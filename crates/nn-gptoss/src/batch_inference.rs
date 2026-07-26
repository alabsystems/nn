// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Batch inference engine for gpt-oss-20b.
//!
//! Manages concurrent request processing with continuous batching,
//! KV cache management, and priority-based scheduling.

use crate::config::GptOssConfig;

// -- Padding strategy ---------------------------------------------------------

/// How to pad variable-length sequences within a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// Pad shorter sequences on the left (common for causal LMs).
    LeftPad,
    /// Pad shorter sequences on the right.
    RightPad,
    /// No padding — sequences must already be equal length.
    NoPad,
}

// -- Request priority ---------------------------------------------------------

/// Priority level for inference requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestPriority {
    /// Highest priority — processed first.
    High,
    /// Default priority.
    Normal,
    /// Lowest priority — processed last.
    Low,
}

impl RequestPriority {
    /// Numeric rank for ordering: higher value = higher priority.
    #[must_use]
    fn rank(self) -> u8 {
        match self {
            Self::High => 2,
            Self::Normal => 1,
            Self::Low => 0,
        }
    }
}

impl PartialOrd for RequestPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RequestPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank().cmp(&other.rank())
    }
}

// -- Finish reason ------------------------------------------------------------

/// Why generation stopped for a request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// Reached the `max_new_tokens` limit.
    MaxTokens,
    /// Generated the end-of-sequence token.
    EosToken,
    /// Generated a configured stop sequence.
    StopSequence,
    /// An error occurred during generation.
    Error(String),
}

// -- Batch config -------------------------------------------------------------

/// Configuration for the batch inference engine.
#[derive(Clone, Debug)]
pub struct BatchConfig {
    /// Maximum number of requests in a single batch.
    pub max_batch_size: usize,
    /// Maximum number of requests waiting in the queue.
    pub max_waiting_requests: usize,
    /// Maximum allowed input + output sequence length.
    pub max_sequence_length: usize,
    /// How to pad sequences within a batch.
    pub padding_strategy: PaddingStrategy,
    /// Whether to schedule by priority (true) or FIFO (false).
    pub priority_scheduling: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_waiting_requests: 32,
            max_sequence_length: 4096,
            padding_strategy: PaddingStrategy::LeftPad,
            priority_scheduling: true,
        }
    }
}

// -- Inference request / response ---------------------------------------------

/// A single inference request submitted to the batch engine.
#[derive(Clone, Debug)]
pub struct InferenceRequest {
    /// Unique request identifier.
    pub id: u64,
    /// Input token IDs (prompt).
    pub token_ids: Vec<usize>,
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Scheduling priority.
    pub priority: RequestPriority,
    /// When the request was created (for FIFO tiebreaking).
    pub created_at: std::time::Instant,
}

/// The result of processing a single inference request.
#[derive(Clone, Debug)]
pub struct InferenceResponse {
    /// Matches the request's `id`.
    pub request_id: u64,
    /// Generated output token IDs (excluding the prompt).
    pub generated_tokens: Vec<usize>,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// Number of tokens in the original prompt.
    pub num_prompt_tokens: usize,
    /// Number of tokens generated.
    pub num_generated_tokens: usize,
}

// -- Batch errors -------------------------------------------------------------

/// Errors from batch inference operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BatchError {
    /// The waiting queue is at capacity.
    #[error("queue full: {current}/{max} requests")]
    QueueFull { current: usize, max: usize },

    /// The input sequence exceeds the configured maximum length.
    #[error("sequence too long: {length} > {max}")]
    SequenceTooLong { length: usize, max: usize },

    /// Generic invalid request.
    #[error("invalid request: {0}")]
    InvalidRequest(String),
}

// -- Batch scheduler ----------------------------------------------------------

/// Schedules inference requests into batches.
///
/// Maintains a priority-ordered waiting queue and yields batches of up to
/// `max_batch_size` requests at a time.
pub struct BatchScheduler {
    /// Pending requests, ordered by priority then arrival time.
    waiting: Vec<InferenceRequest>,
    /// Engine configuration.
    config: BatchConfig,
}

impl BatchScheduler {
    /// Create a new scheduler with the given configuration.
    #[must_use]
    pub fn new(config: BatchConfig) -> Self {
        Self {
            waiting: Vec::new(),
            config,
        }
    }

    /// Submit a new request to the waiting queue.
    ///
    /// Returns `Err(BatchError::QueueFull)` if the queue is at capacity,
    /// or `Err(BatchError::SequenceTooLong)` if the input exceeds the
    /// configured maximum.
    pub fn submit(&mut self, request: InferenceRequest) -> Result<(), BatchError> {
        if self.waiting.len() >= self.config.max_waiting_requests {
            return Err(BatchError::QueueFull {
                current: self.waiting.len(),
                max: self.config.max_waiting_requests,
            });
        }
        if request.token_ids.len() > self.config.max_sequence_length {
            return Err(BatchError::SequenceTooLong {
                length: request.token_ids.len(),
                max: self.config.max_sequence_length,
            });
        }
        if request.token_ids.is_empty() {
            return Err(BatchError::InvalidRequest(
                "token_ids must not be empty".into(),
            ));
        }
        self.waiting.push(request);
        Ok(())
    }

    /// Select the next batch of up to `max_batch_size` requests.
    ///
    /// When `priority_scheduling` is enabled, higher-priority requests are
    /// selected first (with FIFO tiebreaking within the same priority).
    /// Returns `None` if the queue is empty.
    pub fn next_batch(&mut self) -> Option<Vec<InferenceRequest>> {
        if self.waiting.is_empty() {
            return None;
        }

        if self.config.priority_scheduling {
            // Sort: highest priority first, then earliest created_at first.
            self.waiting.sort_by(|a, b| {
                b.priority
                    .cmp(&a.priority)
                    .then_with(|| a.created_at.cmp(&b.created_at))
            });
        }
        // Otherwise FIFO: order of insertion is preserved.

        let batch_size = self.waiting.len().min(self.config.max_batch_size);
        let batch: Vec<InferenceRequest> = self.waiting.drain(..batch_size).collect();
        Some(batch)
    }

    /// Number of requests currently waiting.
    #[must_use]
    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    /// Whether the waiting queue is at capacity.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.waiting.len() >= self.config.max_waiting_requests
    }

    /// Reference to the engine configuration.
    #[must_use]
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }
}

// -- Padding ------------------------------------------------------------------

/// Pad a set of variable-length token sequences to a uniform length.
///
/// Returns `(padded_sequences, original_lengths)` where each padded sequence
/// has the same length (the maximum in the batch). The padding token is
/// `pad_id`.
///
/// With `NoPad`, sequences are returned unchanged (caller must ensure they
/// are already equal length).
#[must_use]
pub fn pad_sequences(
    sequences: &[Vec<usize>],
    strategy: PaddingStrategy,
    pad_id: usize,
) -> (Vec<Vec<usize>>, Vec<usize>) {
    if sequences.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let original_lengths: Vec<usize> = sequences.iter().map(Vec::len).collect();
    let max_len = original_lengths.iter().copied().max().unwrap_or(0);

    let padded = match strategy {
        PaddingStrategy::NoPad => sequences.to_vec(),
        PaddingStrategy::LeftPad => sequences
            .iter()
            .map(|seq| {
                let pad_count = max_len.saturating_sub(seq.len());
                let mut padded = vec![pad_id; pad_count];
                padded.extend_from_slice(seq);
                padded
            })
            .collect(),
        PaddingStrategy::RightPad => sequences
            .iter()
            .map(|seq| {
                let mut padded = seq.clone();
                padded.resize(max_len, pad_id);
                padded
            })
            .collect(),
    };

    (padded, original_lengths)
}

// -- Memory estimation --------------------------------------------------------

/// Estimate total memory (bytes) for batch inference.
///
/// Accounts for:
/// - Activation memory: `batch_size * max_seq * hidden_size * 4` (F32)
/// - KV cache: `2 * num_layers * batch_size * num_kv_heads * max_seq * head_dim * 4`
///
/// Returns `None` if arithmetic overflows.
#[must_use]
pub fn estimate_batch_memory(
    cfg: &GptOssConfig,
    batch_size: usize,
    max_seq: usize,
) -> Option<usize> {
    let bpe = 4_usize; // F32

    // Activation memory: batch_size * max_seq * hidden_size * bpe
    let activation = batch_size
        .checked_mul(max_seq)?
        .checked_mul(cfg.hidden_size)?
        .checked_mul(bpe)?;

    // KV cache: 2 * num_layers * batch_size * num_kv_heads * max_seq * head_dim * bpe
    let kv_cache = 2_usize
        .checked_mul(cfg.num_hidden_layers)?
        .checked_mul(batch_size)?
        .checked_mul(cfg.num_key_value_heads)?
        .checked_mul(max_seq)?
        .checked_mul(cfg.head_dim)?
        .checked_mul(bpe)?;

    activation.checked_add(kv_cache)
}

// -- Tests --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(id: u64, tokens: usize, priority: RequestPriority) -> InferenceRequest {
        InferenceRequest {
            id,
            token_ids: vec![1; tokens],
            max_new_tokens: 64,
            priority,
            created_at: std::time::Instant::now(),
        }
    }

    // -- BatchConfig defaults -------------------------------------------------

    #[test]
    fn test_batch_config_defaults() {
        let cfg = BatchConfig::default();
        assert_eq!(cfg.max_batch_size, 8);
        assert_eq!(cfg.max_waiting_requests, 32);
        assert_eq!(cfg.max_sequence_length, 4096);
        assert_eq!(cfg.padding_strategy, PaddingStrategy::LeftPad);
        assert!(cfg.priority_scheduling);
    }

    // -- Priority ordering ----------------------------------------------------

    #[test]
    fn test_priority_ordering_high_gt_normal_gt_low() {
        assert!(RequestPriority::High > RequestPriority::Normal);
        assert!(RequestPriority::Normal > RequestPriority::Low);
        assert!(RequestPriority::High > RequestPriority::Low);
    }

    #[test]
    fn test_priority_equality() {
        assert_eq!(RequestPriority::High, RequestPriority::High);
        assert_eq!(RequestPriority::Normal, RequestPriority::Normal);
        assert_eq!(RequestPriority::Low, RequestPriority::Low);
    }

    // -- Scheduler submit -----------------------------------------------------

    #[test]
    fn test_submit_success() {
        let mut scheduler = BatchScheduler::new(BatchConfig::default());
        let req = make_request(1, 10, RequestPriority::Normal);
        assert!(scheduler.submit(req).is_ok());
        assert_eq!(scheduler.waiting_count(), 1);
    }

    #[test]
    fn test_submit_queue_full() {
        let cfg = BatchConfig {
            max_waiting_requests: 2,
            ..BatchConfig::default()
        };
        let mut scheduler = BatchScheduler::new(cfg);
        scheduler
            .submit(make_request(1, 10, RequestPriority::Normal))
            .unwrap();
        scheduler
            .submit(make_request(2, 10, RequestPriority::Normal))
            .unwrap();
        let err = scheduler
            .submit(make_request(3, 10, RequestPriority::Normal))
            .unwrap_err();
        assert!(matches!(err, BatchError::QueueFull { .. }));
    }

    #[test]
    fn test_submit_sequence_too_long() {
        let cfg = BatchConfig {
            max_sequence_length: 100,
            ..BatchConfig::default()
        };
        let mut scheduler = BatchScheduler::new(cfg);
        let err = scheduler
            .submit(make_request(1, 200, RequestPriority::Normal))
            .unwrap_err();
        assert!(matches!(err, BatchError::SequenceTooLong { .. }));
    }

    #[test]
    fn test_submit_empty_tokens() {
        let mut scheduler = BatchScheduler::new(BatchConfig::default());
        let req = InferenceRequest {
            id: 1,
            token_ids: vec![],
            max_new_tokens: 10,
            priority: RequestPriority::Normal,
            created_at: std::time::Instant::now(),
        };
        let err = scheduler.submit(req).unwrap_err();
        assert!(matches!(err, BatchError::InvalidRequest(_)));
    }

    // -- Scheduler next_batch -------------------------------------------------

    #[test]
    fn test_next_batch_empty_returns_none() {
        let mut scheduler = BatchScheduler::new(BatchConfig::default());
        assert!(scheduler.next_batch().is_none());
    }

    #[test]
    fn test_next_batch_respects_max_batch_size() {
        let cfg = BatchConfig {
            max_batch_size: 3,
            ..BatchConfig::default()
        };
        let mut scheduler = BatchScheduler::new(cfg);
        for i in 0..10 {
            scheduler
                .submit(make_request(i, 10, RequestPriority::Normal))
                .unwrap();
        }
        let batch = scheduler.next_batch().unwrap();
        assert_eq!(batch.len(), 3);
        assert_eq!(scheduler.waiting_count(), 7);
    }

    #[test]
    fn test_next_batch_priority_scheduling() {
        let cfg = BatchConfig {
            max_batch_size: 2,
            priority_scheduling: true,
            ..BatchConfig::default()
        };
        let mut scheduler = BatchScheduler::new(cfg);
        scheduler
            .submit(make_request(1, 10, RequestPriority::Low))
            .unwrap();
        scheduler
            .submit(make_request(2, 10, RequestPriority::High))
            .unwrap();
        scheduler
            .submit(make_request(3, 10, RequestPriority::Normal))
            .unwrap();

        let batch = scheduler.next_batch().unwrap();
        assert_eq!(batch.len(), 2);
        // High priority (id=2) should come first, then Normal (id=3)
        assert_eq!(batch[0].id, 2);
        assert_eq!(batch[1].id, 3);
    }

    #[test]
    fn test_next_batch_fifo_when_priority_disabled() {
        let cfg = BatchConfig {
            max_batch_size: 2,
            priority_scheduling: false,
            ..BatchConfig::default()
        };
        let mut scheduler = BatchScheduler::new(cfg);
        scheduler
            .submit(make_request(1, 10, RequestPriority::Low))
            .unwrap();
        scheduler
            .submit(make_request(2, 10, RequestPriority::High))
            .unwrap();

        let batch = scheduler.next_batch().unwrap();
        // FIFO: id=1 first regardless of priority
        assert_eq!(batch[0].id, 1);
        assert_eq!(batch[1].id, 2);
    }

    #[test]
    fn test_next_batch_drains_queue() {
        let cfg = BatchConfig {
            max_batch_size: 10,
            ..BatchConfig::default()
        };
        let mut scheduler = BatchScheduler::new(cfg);
        scheduler
            .submit(make_request(1, 10, RequestPriority::Normal))
            .unwrap();
        scheduler
            .submit(make_request(2, 10, RequestPriority::Normal))
            .unwrap();

        let batch = scheduler.next_batch().unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(scheduler.waiting_count(), 0);
        assert!(scheduler.next_batch().is_none());
    }

    // -- is_full --------------------------------------------------------------

    #[test]
    fn test_is_full() {
        let cfg = BatchConfig {
            max_waiting_requests: 2,
            ..BatchConfig::default()
        };
        let mut scheduler = BatchScheduler::new(cfg);
        assert!(!scheduler.is_full());
        scheduler
            .submit(make_request(1, 10, RequestPriority::Normal))
            .unwrap();
        assert!(!scheduler.is_full());
        scheduler
            .submit(make_request(2, 10, RequestPriority::Normal))
            .unwrap();
        assert!(scheduler.is_full());
    }

    // -- Padding --------------------------------------------------------------

    #[test]
    fn test_pad_sequences_left_pad() {
        let seqs = vec![vec![1, 2, 3], vec![4, 5]];
        let (padded, lengths) = pad_sequences(&seqs, PaddingStrategy::LeftPad, 0);
        assert_eq!(lengths, vec![3, 2]);
        assert_eq!(padded[0], vec![1, 2, 3]);
        assert_eq!(padded[1], vec![0, 4, 5]);
    }

    #[test]
    fn test_pad_sequences_right_pad() {
        let seqs = vec![vec![1, 2, 3], vec![4, 5]];
        let (padded, lengths) = pad_sequences(&seqs, PaddingStrategy::RightPad, 0);
        assert_eq!(lengths, vec![3, 2]);
        assert_eq!(padded[0], vec![1, 2, 3]);
        assert_eq!(padded[1], vec![4, 5, 0]);
    }

    #[test]
    fn test_pad_sequences_no_pad() {
        let seqs = vec![vec![1, 2], vec![3, 4]];
        let (padded, lengths) = pad_sequences(&seqs, PaddingStrategy::NoPad, 0);
        assert_eq!(lengths, vec![2, 2]);
        assert_eq!(padded, seqs);
    }

    #[test]
    fn test_pad_sequences_empty_input() {
        let seqs: Vec<Vec<usize>> = vec![];
        let (padded, lengths) = pad_sequences(&seqs, PaddingStrategy::LeftPad, 0);
        assert!(padded.is_empty());
        assert!(lengths.is_empty());
    }

    #[test]
    fn test_pad_sequences_single_sequence() {
        let seqs = vec![vec![10, 20, 30]];
        let (padded, lengths) = pad_sequences(&seqs, PaddingStrategy::LeftPad, 0);
        assert_eq!(lengths, vec![3]);
        assert_eq!(padded[0], vec![10, 20, 30]);
    }

    #[test]
    fn test_pad_preserves_original_content() {
        let seqs = vec![vec![7, 8], vec![9, 10, 11]];
        let (padded, _) = pad_sequences(&seqs, PaddingStrategy::LeftPad, 0);
        // Original tokens of seq[0] should appear at the end (left-padded)
        assert_eq!(&padded[0][1..], &[7, 8]);
        // Original tokens of seq[1] are unchanged (max length)
        assert_eq!(padded[1], vec![9, 10, 11]);
    }

    // -- Memory estimation ----------------------------------------------------

    #[test]
    fn test_estimate_batch_memory_nonzero() {
        let cfg = GptOssConfig::gptoss_20b();
        let mem = estimate_batch_memory(&cfg, 4, 512).expect("should not overflow");
        assert!(mem > 0);
    }

    #[test]
    fn test_estimate_batch_memory_zero_batch() {
        let cfg = GptOssConfig::gptoss_20b();
        let mem = estimate_batch_memory(&cfg, 0, 512).expect("should not overflow");
        assert_eq!(mem, 0);
    }

    #[test]
    fn test_estimate_batch_memory_zero_seq() {
        let cfg = GptOssConfig::gptoss_20b();
        let mem = estimate_batch_memory(&cfg, 4, 0).expect("should not overflow");
        assert_eq!(mem, 0);
    }

    #[test]
    fn test_estimate_batch_memory_monotonic_in_batch() {
        let cfg = GptOssConfig::gptoss_20b();
        let mem_4 = estimate_batch_memory(&cfg, 4, 512).unwrap();
        let mem_8 = estimate_batch_memory(&cfg, 8, 512).unwrap();
        assert!(mem_8 > mem_4, "larger batch should need more memory");
    }

    #[test]
    fn test_estimate_batch_memory_monotonic_in_seq() {
        let cfg = GptOssConfig::gptoss_20b();
        let mem_256 = estimate_batch_memory(&cfg, 4, 256).unwrap();
        let mem_512 = estimate_batch_memory(&cfg, 4, 512).unwrap();
        assert!(mem_512 > mem_256, "longer seq should need more memory");
    }

    // -- InferenceResponse construction ---------------------------------------

    #[test]
    fn test_inference_response_fields() {
        let resp = InferenceResponse {
            request_id: 42,
            generated_tokens: vec![100, 200, 300],
            finish_reason: FinishReason::EosToken,
            num_prompt_tokens: 10,
            num_generated_tokens: 3,
        };
        assert_eq!(resp.request_id, 42);
        assert_eq!(resp.generated_tokens.len(), 3);
        assert_eq!(resp.finish_reason, FinishReason::EosToken);
        assert_eq!(resp.num_prompt_tokens, 10);
        assert_eq!(resp.num_generated_tokens, 3);
    }

    #[test]
    fn test_finish_reason_error_variant() {
        let reason = FinishReason::Error("timeout".into());
        assert_eq!(reason, FinishReason::Error("timeout".into()));
        assert_ne!(reason, FinishReason::MaxTokens);
    }

    // -- Batch error display --------------------------------------------------

    #[test]
    fn test_batch_error_display() {
        let err = BatchError::QueueFull {
            current: 32,
            max: 32,
        };
        assert!(err.to_string().contains("queue full"));

        let err = BatchError::SequenceTooLong {
            length: 5000,
            max: 4096,
        };
        assert!(err.to_string().contains("sequence too long"));

        let err = BatchError::InvalidRequest("bad input".into());
        assert!(err.to_string().contains("bad input"));
    }
}
