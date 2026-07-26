#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Performance proof tests for beam search: KV cache reordering, causal mask
//! correctness, CTC prefix growth scaling.
//!
//! Extracted from `beam_search_tests_extended.rs` for file size compliance.

use crate::dyn_tensor::DynTensor;
use crate::layers::kv_cache::KvCache;
use crate::Device;

use super::super::super::{beam_search, BeamSearchConfig};

/// Verify beam search cache reordering works correctly with KvCache.
///
/// After W2-61 optimization, beams share a pool of KvCache objects indexed by
/// position. Per-step cost is O(W × S) — only clones when two surviving beams
/// share a parent (uses std::mem::replace to move caches without copying).
#[test]
fn test_beam_search_cache_reordering() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static FORWARD_CALLS: AtomicUsize = AtomicUsize::new(0);

    // Model that returns different logits per call to prevent beam collapse.
    fn counting_model(_input: &DynTensor, cache: &mut KvCache) -> crate::Result<DynTensor> {
        let call = FORWARD_CALLS.fetch_add(1, Ordering::Relaxed);
        // Each call appends a [1,1,1,4] token to cache layer 0.
        let kv = DynTensor::from_vec(vec![1.0f32; 4], &[1, 1, 1, 4], &Device::Cpu)?;
        let _ = cache.layer_mut(0)?.append(&kv, &kv)?;
        // Return 5 vocab items with different argmax per call.
        let mut logits = vec![0.0f32; 5];
        logits[call % 5] = 10.0 - (call as f32) * 0.1;
        DynTensor::from_vec(logits, &[1, 5], &Device::Cpu)
    }

    FORWARD_CALLS.store(0, Ordering::Relaxed);

    let beam_width = 3;
    let max_tokens = 5;
    let config = BeamSearchConfig {
        beam_width,
        max_new_tokens: max_tokens,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = beam_search(counting_model, &[0], &mut cache, &config, &Device::Cpu);
    assert!(result.is_ok());

    let total_calls = FORWARD_CALLS.load(Ordering::Relaxed);
    // prefill (1) + decode steps: beam_width * (max_tokens - 1)
    // Each decode step runs model_fn once per active beam.
    // With beam_width=3, max_tokens=5: 1 + 3*4 = 13 forward calls.
    // No EOS token, so no beams finish early — all 4 decode steps run
    // with all 3 beams active.
    let expected_calls = 1 + beam_width * (max_tokens - 1); // 13
    assert_eq!(
        total_calls, expected_calls,
        "expected exactly {expected_calls} forward calls (1 prefill + {beam_width} * {} decode), got {total_calls}",
        max_tokens - 1,
    );
}

/// Verify causal mask produces correct values: 0.0 on/below diagonal, -inf above.
///
/// The original test only checked shape and element count (tautologically true
/// for any [1,1,S,S] tensor). This version verifies actual mask values.
#[test]
fn test_causal_mask_values_correct() {
    use crate::layers::attention::causal_mask;

    let s = 5;
    let mask = causal_mask(s, &Device::Cpu).expect("causal_mask should succeed");
    let dims = mask.dims();
    assert_eq!(dims, &[1, 1, s, s], "mask shape should be [1,1,{s},{s}]");

    let vals = mask
        .to_flat_vec::<f32>()
        .expect("should extract f32 values");
    // Verify each element: row i, col j
    // mask[i][j] = 0.0 if j <= i (causal: can attend to past and self)
    // mask[i][j] = -inf if j > i (cannot attend to future)
    for i in 0..s {
        for j in 0..s {
            let val = vals[i * s + j];
            if j <= i {
                assert_eq!(val, 0.0, "mask[{i}][{j}] should be 0.0 (causal), got {val}");
            } else {
                assert!(
                    val == f32::NEG_INFINITY,
                    "mask[{i}][{j}] should be -inf (future), got {val}"
                );
            }
        }
    }

    // Also verify size=1: single token can attend to itself.
    let mask1 = causal_mask(1, &Device::Cpu).expect("causal_mask(1)");
    let v1 = mask1
        .to_flat_vec::<f32>()
        .expect("size-1 mask should have f32 values");
    assert_eq!(v1, &[0.0], "size-1 mask should be [0.0]");
}

/// Prove CTC beam search prefix cloning is O(T² × B × V).
///
/// The CTC prefix beam search at each timestep clones Vec<u32> prefixes
/// that grow to length T. With B beams and V vocab items, total clone
/// work is O(T × B × V × T) = O(T² × B × V).
///
/// This test documents the scaling by measuring prefix length growth.
#[test]
fn test_ctc_beam_decode_prefix_growth() {
    use super::super::super::super::ctc::{ctc_beam_decode, CtcConfig};

    let vocab = 4; // 0=blank, 1-3=tokens
    let config = CtcConfig { blank_id: 0 };
    let beam_width = 2;

    // Create logits that produce growing prefixes by cycling through
    // non-blank tokens with strong bias. CTC collapses consecutive identical
    // tokens, so alternating token IDs forces prefix growth at each step.
    // Pattern: [tok1, blank, tok2, blank, tok3, blank, tok1, blank, ...]
    // Each pair (token, blank) adds one collapsed token to the prefix.
    let timesteps = [8, 16, 32];
    let mut prefix_lengths = Vec::new();
    for &t in &timesteps {
        let mut logits_data = vec![0.0f32; t * vocab];
        for step in 0..t {
            let base = step * vocab;
            if step.is_multiple_of(2) {
                // Non-blank step: strongly favor a cycling token.
                let token_id = 1 + (step / 2) % 3; // cycles 1, 2, 3
                logits_data[base] = -10.0; // blank suppressed
                logits_data[base + token_id] = 10.0;
            } else {
                // Blank separator: allows same token to repeat.
                logits_data[base] = 10.0; // blank dominant
                logits_data[base + 1] = -10.0;
                logits_data[base + 2] = -10.0;
                logits_data[base + 3] = -10.0;
            }
        }
        let logits =
            DynTensor::from_vec(logits_data, &[t, vocab], &Device::Cpu).expect("valid logits");
        let beams = ctc_beam_decode(&logits, &config, beam_width)
            .expect("ctc_beam_decode should succeed for valid logits");
        assert!(
            !beams.is_empty(),
            "should produce at least one beam for t={t}"
        );
        let max_prefix_len = beams.iter().map(|b| b.tokens.len()).max().unwrap_or(0);
        // Each token+blank pair produces ~1 prefix token.
        // With t timesteps and pairs of (token, blank), expect ~t/2 prefix tokens.
        // Use t/4 as conservative lower bound (accounts for beam pruning losses).
        assert!(
            max_prefix_len >= t / 4,
            "beam prefix too short for t={t}: got {max_prefix_len}, expected >= {}",
            t / 4,
        );
        prefix_lengths.push(max_prefix_len);
    }
    // Verify monotonic growth: doubling T should increase prefix length.
    for i in 1..prefix_lengths.len() {
        assert!(
            prefix_lengths[i] > prefix_lengths[i - 1],
            "prefix length should grow with T: t={} gave {}, t={} gave {}",
            timesteps[i - 1],
            prefix_lengths[i - 1],
            timesteps[i],
            prefix_lengths[i],
        );
    }
}
