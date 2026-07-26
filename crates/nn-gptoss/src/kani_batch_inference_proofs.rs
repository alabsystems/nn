// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proofs for batch inference correctness.
//!
//! Proves:
//! 1. `next_batch` returns at most `max_batch_size` requests
//! 2. `pad_sequences` preserves original token content
//! 3. `RequestPriority::High > Normal > Low` ordering
//! 4. `submit` fails when queue is at `max_waiting_requests`
//! 5. `estimate_batch_memory` is monotonically non-decreasing in batch size

use crate::batch_inference::{
    estimate_batch_memory, pad_sequences, BatchConfig, BatchScheduler, InferenceRequest,
    PaddingStrategy, RequestPriority,
};
use crate::config::{GptOssConfig, LayerType};

/// Helper: build a small valid config for Kani exploration.
fn small_valid_config(hidden: usize, num_layers: usize) -> GptOssConfig {
    let layer_types: Vec<LayerType> = (0..num_layers)
        .map(|i| {
            if i % 2 == 0 {
                LayerType::SlidingAttention
            } else {
                LayerType::FullAttention
            }
        })
        .collect();

    GptOssConfig::new(
        hidden,
        hidden,
        num_layers,
        2, // num_attention_heads
        2, // num_key_value_heads
        4, // head_dim
        8, // vocab_size
        1e-5,
        150_000.0,
        4096,
        false,
        None,
        true,
        2, // num_local_experts
        1, // experts_per_token
        7.0,
        layer_types,
        128,
        200_002,
    )
}

/// Helper: create a request with the given token count.
fn make_request(id: u64, num_tokens: usize, priority: RequestPriority) -> InferenceRequest {
    InferenceRequest {
        id,
        token_ids: vec![1; num_tokens],
        max_new_tokens: 10,
        priority,
        created_at: std::time::Instant::now(),
    }
}

// ============================================================================
// Proof 1: next_batch returns at most max_batch_size requests
// ============================================================================

/// For any number of submitted requests, `next_batch` never returns more
/// than `max_batch_size` elements.
#[kani::proof]
fn proof_batch_size_bounded() {
    let max_batch: usize = kani::any();
    kani::assume(max_batch >= 1 && max_batch <= 8);

    let num_requests: usize = kani::any();
    kani::assume(num_requests >= 1 && num_requests <= 16);

    let cfg = BatchConfig {
        max_batch_size: max_batch,
        max_waiting_requests: 32,
        max_sequence_length: 4096,
        padding_strategy: PaddingStrategy::LeftPad,
        priority_scheduling: false,
    };
    let mut scheduler = BatchScheduler::new(cfg);

    for i in 0..num_requests {
        let _ = scheduler.submit(make_request(i as u64, 5, RequestPriority::Normal));
    }

    if let Some(batch) = scheduler.next_batch() {
        assert!(
            batch.len() <= max_batch,
            "batch must not exceed max_batch_size"
        );
    }
}

// ============================================================================
// Proof 2: pad_sequences preserves original token content
// ============================================================================

/// After left-padding, the original tokens of each sequence appear at the
/// end of the padded sequence and are unchanged.
#[kani::proof]
fn proof_padding_preserves_content() {
    // Use two small sequences with symbolic lengths
    let len_a: usize = kani::any();
    let len_b: usize = kani::any();
    kani::assume(len_a >= 1 && len_a <= 4);
    kani::assume(len_b >= 1 && len_b <= 4);

    let seq_a: Vec<usize> = vec![42; len_a];
    let seq_b: Vec<usize> = vec![99; len_b];
    let seqs = vec![seq_a.clone(), seq_b.clone()];
    let pad_id = 0_usize;

    let (padded, lengths) = pad_sequences(&seqs, PaddingStrategy::LeftPad, pad_id);

    assert_eq!(lengths[0], len_a);
    assert_eq!(lengths[1], len_b);

    let max_len = len_a.max(len_b);
    assert_eq!(padded[0].len(), max_len);
    assert_eq!(padded[1].len(), max_len);

    // Original content of seq_a appears at the tail
    for i in 0..len_a {
        assert_eq!(padded[0][max_len - len_a + i], 42);
    }
    // Original content of seq_b appears at the tail
    for i in 0..len_b {
        assert_eq!(padded[1][max_len - len_b + i], 99);
    }
}

// ============================================================================
// Proof 3: priority ordering (High > Normal > Low)
// ============================================================================

/// The `Ord` implementation on `RequestPriority` satisfies High > Normal > Low.
#[kani::proof]
fn proof_priority_ordering() {
    assert!(RequestPriority::High > RequestPriority::Normal);
    assert!(RequestPriority::Normal > RequestPriority::Low);
    assert!(RequestPriority::High > RequestPriority::Low);

    // Reflexive: each priority is equal to itself
    assert!(RequestPriority::High == RequestPriority::High);
    assert!(RequestPriority::Normal == RequestPriority::Normal);
    assert!(RequestPriority::Low == RequestPriority::Low);

    // Antisymmetric: not (Normal > High)
    assert!(!(RequestPriority::Normal > RequestPriority::High));
    assert!(!(RequestPriority::Low > RequestPriority::Normal));
}

// ============================================================================
// Proof 4: submit fails when queue is at max_waiting_requests
// ============================================================================

/// When exactly `max_waiting_requests` are in the queue, the next submit
/// returns `Err(QueueFull)`.
#[kani::proof]
fn proof_queue_full_rejection() {
    let max_waiting: usize = kani::any();
    kani::assume(max_waiting >= 1 && max_waiting <= 8);

    let cfg = BatchConfig {
        max_batch_size: 4,
        max_waiting_requests: max_waiting,
        max_sequence_length: 4096,
        padding_strategy: PaddingStrategy::LeftPad,
        priority_scheduling: false,
    };
    let mut scheduler = BatchScheduler::new(cfg);

    // Fill the queue to capacity
    for i in 0..max_waiting {
        let result = scheduler.submit(make_request(i as u64, 5, RequestPriority::Normal));
        assert!(result.is_ok(), "should accept request {i} before capacity");
    }

    assert_eq!(scheduler.waiting_count(), max_waiting);
    assert!(scheduler.is_full());

    // The next submit must fail
    let result = scheduler.submit(make_request(max_waiting as u64, 5, RequestPriority::Normal));
    assert!(result.is_err(), "must reject request when queue is full");
}

// ============================================================================
// Proof 5: estimate_batch_memory is monotonically non-decreasing in batch_size
// ============================================================================

/// For a fixed sequence length, increasing batch_size never decreases
/// estimated memory.
#[kani::proof]
fn proof_batch_memory_monotonic() {
    let batch_a: usize = kani::any();
    let batch_b: usize = kani::any();
    kani::assume(batch_a >= 1 && batch_a <= 8);
    kani::assume(batch_b >= batch_a && batch_b <= 16);

    let seq_len: usize = kani::any();
    kani::assume(seq_len >= 1 && seq_len <= 64);

    let cfg = small_valid_config(4, 2);

    if let (Some(mem_a), Some(mem_b)) = (
        estimate_batch_memory(&cfg, batch_a, seq_len),
        estimate_batch_memory(&cfg, batch_b, seq_len),
    ) {
        assert!(
            mem_b >= mem_a,
            "batch memory must be monotonically non-decreasing in batch_size"
        );
    }
}
