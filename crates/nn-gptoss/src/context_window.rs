// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Context window management for gpt-oss 131K-token sequences.
//!
//! Manages position tracking, sliding window eviction for attention layers,
//! and chunked processing for long documents. The model uses YaRN RoPE
//! to extend effective context from 4096 to 131,072 tokens.

// -- Configuration -----------------------------------------------------------

/// Configuration for context window behavior.
///
/// Controls the maximum context length, YaRN scaling parameters, sliding
/// window size for local attention layers, and prefill chunking granularity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextWindowConfig {
    /// Maximum context length in tokens.
    ///
    /// For gpt-oss-20b with YaRN this is 131,072.
    pub max_context_length: usize,

    /// Original maximum position before YaRN extension.
    ///
    /// The base RoPE was trained for this many positions (4096 for gpt-oss).
    /// YaRN stretches the position space to `max_context_length`.
    pub original_max_position: usize,

    /// Sliding window size for local attention layers.
    ///
    /// Even-numbered layers in gpt-oss use sliding window attention limited
    /// to this many tokens (128 for the default configuration).
    pub sliding_window_size: usize,

    /// Chunk size for prefill processing.
    ///
    /// Long prompts are split into chunks of this size to bound peak memory
    /// during the prefill phase. Must be > 0.
    pub prefill_chunk_size: usize,

    /// Whether YaRN RoPE scaling is enabled.
    ///
    /// When true, positions beyond `original_max_position` are scaled
    /// according to the YaRN factor.
    pub yarn_enabled: bool,
}

impl Default for ContextWindowConfig {
    fn default() -> Self {
        Self {
            max_context_length: 131_072,
            original_max_position: 4096,
            sliding_window_size: 128,
            prefill_chunk_size: 4096,
            yarn_enabled: true,
        }
    }
}

impl ContextWindowConfig {
    /// Create a new configuration with all fields specified.
    #[must_use]
    pub fn new(
        max_context_length: usize,
        original_max_position: usize,
        sliding_window_size: usize,
        prefill_chunk_size: usize,
        yarn_enabled: bool,
    ) -> Self {
        Self {
            max_context_length,
            original_max_position,
            sliding_window_size,
            prefill_chunk_size,
            yarn_enabled,
        }
    }
}

// -- Context Window ----------------------------------------------------------

/// Tracks current position state within the context window.
///
/// Maintains the current token position and total tokens processed,
/// supporting advance, reset, and capacity queries. Used by the generation
/// loop to manage position IDs passed to the model and to detect when
/// chunked prefill or context eviction is needed.
#[derive(Clone, Debug)]
pub struct ContextWindow {
    /// Next token position (0-based).
    current_position: usize,

    /// Total tokens processed since creation (survives resets).
    total_tokens_processed: usize,

    /// Configuration for this context window.
    config: ContextWindowConfig,
}

impl ContextWindow {
    /// Create a new context window with the given configuration.
    #[must_use]
    pub fn new(config: ContextWindowConfig) -> Self {
        Self {
            current_position: 0,
            total_tokens_processed: 0,
            config,
        }
    }

    /// Create a context window with the default 131K configuration.
    #[must_use]
    pub fn default_131k() -> Self {
        Self::new(ContextWindowConfig::default())
    }

    /// Advance the position by `n_tokens`.
    ///
    /// Saturates at `max_context_length` to prevent overflow when the context
    /// is exhausted. `total_tokens_processed` always advances (it tracks
    /// lifetime throughput, not position).
    pub fn advance(&mut self, n_tokens: usize) {
        let max = self.config.max_context_length;
        self.current_position = self.current_position.saturating_add(n_tokens).min(max);
        self.total_tokens_processed = self.total_tokens_processed.saturating_add(n_tokens);
    }

    /// Generate position indices for the next `n_tokens`.
    ///
    /// Returns a vector `[current_position, current_position + 1, ...,
    /// current_position + n_tokens - 1]`. Does NOT advance the position;
    /// call `advance()` after the tokens have been processed.
    ///
    /// Returns an empty vec when `n_tokens == 0`.
    #[must_use]
    pub fn positions_for_tokens(&self, n_tokens: usize) -> Vec<usize> {
        (0..n_tokens)
            .map(|i| self.current_position.saturating_add(i))
            .collect()
    }

    /// Remaining capacity: tokens until the context window is exhausted.
    ///
    /// Returns 0 when `current_position >= max_context_length`.
    #[must_use]
    pub fn remaining_capacity(&self) -> usize {
        self.config
            .max_context_length
            .saturating_sub(self.current_position)
    }

    /// Whether the context window is fully exhausted.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.current_position >= self.config.max_context_length
    }

    /// Reset position to the beginning.
    ///
    /// `total_tokens_processed` is preserved across resets so callers can
    /// track lifetime throughput.
    pub fn reset(&mut self) {
        self.current_position = 0;
    }

    /// Whether the given input length requires chunked prefill.
    ///
    /// Returns `true` when `input_len > prefill_chunk_size`.
    #[must_use]
    pub fn needs_chunking(&self, input_len: usize) -> bool {
        input_len > self.config.prefill_chunk_size
    }

    /// Current position (0-based, next token to be generated).
    #[must_use]
    pub fn current_position(&self) -> usize {
        self.current_position
    }

    /// Total tokens processed since creation (lifetime counter).
    #[must_use]
    pub fn total_tokens_processed(&self) -> usize {
        self.total_tokens_processed
    }

    /// Reference to the underlying configuration.
    #[must_use]
    pub fn config(&self) -> &ContextWindowConfig {
        &self.config
    }
}

// -- Free Functions ----------------------------------------------------------

/// Split a position range `[0, total_len)` into chunks of `chunk_size`.
///
/// Returns `(start, len)` pairs. The last chunk may be shorter than
/// `chunk_size`. Returns an empty vec when `total_len == 0`.
///
/// # Panics
///
/// Panics if `chunk_size == 0`.
#[must_use]
pub(crate) fn chunk_positions(total_len: usize, chunk_size: usize) -> Vec<(usize, usize)> {
    assert!(chunk_size > 0, "chunk_size must be > 0");
    if total_len == 0 {
        return Vec::new();
    }

    let full_chunks = total_len / chunk_size;
    let remainder = total_len % chunk_size;
    let n_chunks = full_chunks + usize::from(remainder > 0);

    let mut chunks = Vec::with_capacity(n_chunks);
    for i in 0..full_chunks {
        chunks.push((i * chunk_size, chunk_size));
    }
    if remainder > 0 {
        chunks.push((full_chunks * chunk_size, remainder));
    }
    chunks
}

/// Compute the valid attention range for a sliding window layer.
///
/// Given a token at `position` and a `window_size`, returns `(start, end)`
/// where `start..end` is the range of positions this token can attend to.
///
/// - `start = position.saturating_sub(window_size)`
/// - `end = position + 1` (inclusive of the current position)
///
/// The returned range always satisfies `start <= position < end`.
#[must_use]
pub(crate) fn sliding_window_range(position: usize, window_size: usize) -> (usize, usize) {
    let start = position.saturating_sub(window_size);
    let end = position + 1;
    (start, end)
}

/// Compute the effective "attention distance" after YaRN scaling.
///
/// Positions within `original_max` are unscaled (distance = position).
/// Positions beyond `original_max` have their excess scaled down by
/// `yarn_factor`, compressing the perceived distance so the model can
/// attend further than its original training range.
///
/// Returns `f64` because the YaRN-scaled distance is fractional.
///
/// # Formula
///
/// ```text
/// if position <= original_max:
///     effective = position as f64
/// else:
///     excess = position - original_max
///     effective = original_max as f64 + (excess as f64 / yarn_factor)
/// ```
#[must_use]
pub(crate) fn effective_context_with_yarn(
    position: usize,
    original_max: usize,
    yarn_factor: f64,
) -> f64 {
    if position <= original_max {
        position as f64
    } else {
        let excess = (position - original_max) as f64;
        original_max as f64 + excess / yarn_factor
    }
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ContextWindowConfig ----

    #[test]
    fn test_config_defaults() {
        let cfg = ContextWindowConfig::default();
        assert_eq!(cfg.max_context_length, 131_072);
        assert_eq!(cfg.original_max_position, 4096);
        assert_eq!(cfg.sliding_window_size, 128);
        assert_eq!(cfg.prefill_chunk_size, 4096);
        assert!(cfg.yarn_enabled);
    }

    #[test]
    fn test_config_custom() {
        let cfg = ContextWindowConfig::new(65536, 2048, 64, 1024, false);
        assert_eq!(cfg.max_context_length, 65536);
        assert_eq!(cfg.original_max_position, 2048);
        assert_eq!(cfg.sliding_window_size, 64);
        assert_eq!(cfg.prefill_chunk_size, 1024);
        assert!(!cfg.yarn_enabled);
    }

    // ---- ContextWindow construction & accessors ----

    #[test]
    fn test_new_context_window() {
        let cw = ContextWindow::default_131k();
        assert_eq!(cw.current_position(), 0);
        assert_eq!(cw.total_tokens_processed(), 0);
        assert_eq!(cw.remaining_capacity(), 131_072);
        assert!(!cw.is_full());
    }

    // ---- advance ----

    #[test]
    fn test_advance_basic() {
        let mut cw = ContextWindow::default_131k();
        cw.advance(10);
        assert_eq!(cw.current_position(), 10);
        assert_eq!(cw.total_tokens_processed(), 10);
    }

    #[test]
    fn test_advance_multiple() {
        let mut cw = ContextWindow::default_131k();
        cw.advance(100);
        cw.advance(200);
        assert_eq!(cw.current_position(), 300);
        assert_eq!(cw.total_tokens_processed(), 300);
    }

    #[test]
    fn test_advance_saturates_at_max() {
        let cfg = ContextWindowConfig::new(100, 50, 10, 20, true);
        let mut cw = ContextWindow::new(cfg);
        cw.advance(150);
        assert_eq!(cw.current_position(), 100);
        assert!(cw.is_full());
        assert_eq!(cw.total_tokens_processed(), 150);
    }

    #[test]
    fn test_advance_zero() {
        let mut cw = ContextWindow::default_131k();
        cw.advance(42);
        cw.advance(0);
        assert_eq!(cw.current_position(), 42);
        assert_eq!(cw.total_tokens_processed(), 42);
    }

    // ---- positions_for_tokens ----

    #[test]
    fn test_positions_for_tokens_basic() {
        let mut cw = ContextWindow::default_131k();
        cw.advance(5);
        let positions = cw.positions_for_tokens(3);
        assert_eq!(positions, vec![5, 6, 7]);
    }

    #[test]
    fn test_positions_for_tokens_from_zero() {
        let cw = ContextWindow::default_131k();
        let positions = cw.positions_for_tokens(4);
        assert_eq!(positions, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_positions_for_tokens_empty() {
        let cw = ContextWindow::default_131k();
        let positions = cw.positions_for_tokens(0);
        assert!(positions.is_empty());
    }

    #[test]
    fn test_positions_does_not_advance() {
        let cw = ContextWindow::default_131k();
        let _ = cw.positions_for_tokens(10);
        assert_eq!(cw.current_position(), 0);
    }

    // ---- remaining_capacity ----

    #[test]
    fn test_remaining_capacity_full() {
        let cw = ContextWindow::default_131k();
        assert_eq!(cw.remaining_capacity(), 131_072);
    }

    #[test]
    fn test_remaining_capacity_partial() {
        let mut cw = ContextWindow::default_131k();
        cw.advance(1000);
        assert_eq!(cw.remaining_capacity(), 131_072 - 1000);
    }

    #[test]
    fn test_remaining_capacity_exhausted() {
        let cfg = ContextWindowConfig::new(50, 50, 10, 10, false);
        let mut cw = ContextWindow::new(cfg);
        cw.advance(50);
        assert_eq!(cw.remaining_capacity(), 0);
        assert!(cw.is_full());
    }

    // ---- reset ----

    #[test]
    fn test_reset_clears_position() {
        let mut cw = ContextWindow::default_131k();
        cw.advance(500);
        cw.reset();
        assert_eq!(cw.current_position(), 0);
        assert_eq!(cw.remaining_capacity(), 131_072);
        assert!(!cw.is_full());
    }

    #[test]
    fn test_reset_preserves_total() {
        let mut cw = ContextWindow::default_131k();
        cw.advance(500);
        cw.reset();
        assert_eq!(cw.total_tokens_processed(), 500);
    }

    #[test]
    fn test_reset_then_advance() {
        let mut cw = ContextWindow::default_131k();
        cw.advance(100);
        cw.reset();
        cw.advance(30);
        assert_eq!(cw.current_position(), 30);
        assert_eq!(cw.total_tokens_processed(), 130);
    }

    // ---- needs_chunking ----

    #[test]
    fn test_needs_chunking_small_input() {
        let cw = ContextWindow::default_131k();
        assert!(!cw.needs_chunking(100));
    }

    #[test]
    fn test_needs_chunking_exact_boundary() {
        let cw = ContextWindow::default_131k();
        // prefill_chunk_size = 4096; exact match does NOT need chunking
        assert!(!cw.needs_chunking(4096));
    }

    #[test]
    fn test_needs_chunking_over_boundary() {
        let cw = ContextWindow::default_131k();
        assert!(cw.needs_chunking(4097));
    }

    // ---- chunk_positions ----

    #[test]
    fn test_chunk_positions_exact_multiple() {
        let chunks = chunk_positions(12, 4);
        assert_eq!(chunks, vec![(0, 4), (4, 4), (8, 4)]);
    }

    #[test]
    fn test_chunk_positions_with_remainder() {
        let chunks = chunk_positions(10, 4);
        assert_eq!(chunks, vec![(0, 4), (4, 4), (8, 2)]);
    }

    #[test]
    fn test_chunk_positions_single_chunk() {
        let chunks = chunk_positions(3, 10);
        assert_eq!(chunks, vec![(0, 3)]);
    }

    #[test]
    fn test_chunk_positions_single_element() {
        let chunks = chunk_positions(1, 1);
        assert_eq!(chunks, vec![(0, 1)]);
    }

    #[test]
    fn test_chunk_positions_empty() {
        let chunks = chunk_positions(0, 4);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_positions_cover_full_range() {
        let total = 100;
        let chunks = chunk_positions(total, 7);
        // Verify contiguous coverage
        let mut covered = 0;
        for (start, len) in &chunks {
            assert_eq!(*start, covered, "chunk start must equal previous end");
            covered += len;
        }
        assert_eq!(covered, total, "chunks must cover the full range");
    }

    #[test]
    #[should_panic(expected = "chunk_size must be > 0")]
    fn test_chunk_positions_zero_chunk_panics() {
        let _ = chunk_positions(10, 0);
    }

    // ---- sliding_window_range ----

    #[test]
    fn test_sliding_window_range_at_start() {
        let (start, end) = sliding_window_range(0, 128);
        assert_eq!(start, 0);
        assert_eq!(end, 1);
    }

    #[test]
    fn test_sliding_window_range_within_window() {
        let (start, end) = sliding_window_range(50, 128);
        assert_eq!(start, 0);
        assert_eq!(end, 51);
    }

    #[test]
    fn test_sliding_window_range_beyond_window() {
        let (start, end) = sliding_window_range(200, 128);
        assert_eq!(start, 72);
        assert_eq!(end, 201);
    }

    #[test]
    fn test_sliding_window_range_contains_position() {
        for pos in [0, 1, 50, 127, 128, 500, 131_071] {
            let (start, end) = sliding_window_range(pos, 128);
            assert!(start <= pos, "start ({start}) must be <= position ({pos})");
            assert!(end > pos, "end ({end}) must be > position ({pos})");
        }
    }

    #[test]
    fn test_sliding_window_range_size() {
        let window = 128;
        // When position >= window, the range size is exactly window + 1
        let (start, end) = sliding_window_range(500, window);
        assert_eq!(end - start, window + 1);
    }

    // ---- effective_context_with_yarn ----

    #[test]
    fn test_yarn_within_original_max() {
        let eff = effective_context_with_yarn(100, 4096, 32.0);
        assert!((eff - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_yarn_at_original_max() {
        let eff = effective_context_with_yarn(4096, 4096, 32.0);
        assert!((eff - 4096.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_yarn_beyond_original_max() {
        // position=4096+3200, excess=3200, yarn_factor=32 -> scaled excess = 100
        let eff = effective_context_with_yarn(4096 + 3200, 4096, 32.0);
        assert!((eff - (4096.0 + 100.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_yarn_at_131k() {
        // position=131072, excess=131072-4096=126976, factor=32
        // effective = 4096 + 126976/32 = 4096 + 3968 = 8064
        let eff = effective_context_with_yarn(131_072, 4096, 32.0);
        assert!((eff - 8064.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_yarn_factor_one() {
        // factor=1 means no compression
        let eff = effective_context_with_yarn(10_000, 4096, 1.0);
        assert!((eff - 10_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_yarn_position_zero() {
        let eff = effective_context_with_yarn(0, 4096, 32.0);
        assert!((eff - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_yarn_monotonic() {
        // Effective distance must be monotonically increasing with position
        let mut prev = 0.0_f64;
        for pos in (0..=131_072).step_by(1000) {
            let eff = effective_context_with_yarn(pos, 4096, 32.0);
            assert!(
                eff >= prev,
                "effective context must be monotonic: {eff} < {prev} at pos {pos}"
            );
            prev = eff;
        }
    }

    // ---- Integration: ContextWindow + chunk_positions ----

    #[test]
    fn test_chunked_prefill_workflow() {
        let cfg = ContextWindowConfig::new(131_072, 4096, 128, 512, true);
        let mut cw = ContextWindow::new(cfg);

        let prompt_len = 2000;
        assert!(cw.needs_chunking(prompt_len));

        let chunks = chunk_positions(prompt_len, 512);
        assert_eq!(chunks.len(), 4); // 512+512+512+464

        for (start, len) in &chunks {
            let positions = cw.positions_for_tokens(*len);
            assert_eq!(positions[0], *start);
            assert_eq!(positions.len(), *len);
            cw.advance(*len);
        }

        assert_eq!(cw.current_position(), 2000);
        assert_eq!(cw.total_tokens_processed(), 2000);
    }
}
