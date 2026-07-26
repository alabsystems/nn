// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the GPU-resident KV cache decode loop.

use crate::dyn_tensor::DynTensor;
use crate::layers::generation::autoregressive::GenerationConfig;
use crate::layers::generation::decode_loop::{decode_generate, decode_step, prefill, DecodeContext};
use crate::layers::generation::{KvCache, KvCacheBackend};
use crate::{DType, Device};

/// Dummy model: returns logits shaped [1, vocab_size] where token at
/// position `next_token_id` has the highest logit.
///
/// The model tracks the cache via the KvCacheBackend trait: it appends
/// a dummy K/V tensor per layer on each call, simulating real attention.
fn make_dummy_model(
    num_layers: usize,
    vocab_size: usize,
    next_tokens: Vec<usize>,
) -> impl Fn(&DynTensor, &mut KvCache) -> crate::Result<DynTensor> {
    use std::cell::Cell;
    let call_count = Cell::new(0usize);

    move |input: &DynTensor, cache: &mut KvCache| -> crate::Result<DynTensor> {
        let seq_len = input.dim(1)?;
        let idx = call_count.get();
        call_count.set(idx + 1);

        // Append dummy K/V for each layer (simulates attention producing KV).
        let num_heads = 4;
        let head_dim = 8;
        for layer_idx in 0..num_layers {
            let k = DynTensor::ones(&[1, num_heads, seq_len, head_dim], DType::F32, &Device::Cpu)?;
            let v = DynTensor::ones(&[1, num_heads, seq_len, head_dim], DType::F32, &Device::Cpu)?;
            let layer = cache.layer_backend_mut(layer_idx)?;
            layer.append(&k, &v)?;
        }

        // Return logits: highest value at the next_tokens[idx] position.
        let token_idx = if idx < next_tokens.len() {
            next_tokens[idx]
        } else {
            0 // default
        };

        let mut logits_data = vec![0.0f32; vocab_size];
        if token_idx < vocab_size {
            logits_data[token_idx] = 10.0;
        }
        DynTensor::from_vec(logits_data, &[1, vocab_size], &Device::Cpu)
    }
}

// ---------------------------------------------------------------------------
// DecodeContext tests
// ---------------------------------------------------------------------------

#[test]
fn test_decode_context_new() {
    let cache = KvCache::new(4);
    let ctx = DecodeContext::new(cache, 2048);
    assert_eq!(ctx.seq_len(), 0);
    assert_eq!(ctx.generated_count(), 0);
    assert_eq!(ctx.max_seq_len(), 2048);
    assert_eq!(ctx.remaining_capacity(), 2048);
    assert!(!ctx.is_full());
    assert_eq!(ctx.num_layers(), 4);
}

#[test]
fn test_decode_context_reset() {
    let cache = KvCache::new(2);
    let mut ctx = DecodeContext::new(cache, 128);

    // Simulate some generation state.
    ctx.generated_count = 10;
    ctx.reset();
    assert_eq!(ctx.generated_count(), 0);
    assert_eq!(ctx.seq_len(), 0);
}

#[test]
fn test_decode_context_clear() {
    let cache = KvCache::new(2);
    let mut ctx = DecodeContext::new(cache, 128);
    ctx.generated_count = 5;
    ctx.clear();
    assert_eq!(ctx.generated_count(), 0);
    assert_eq!(ctx.seq_len(), 0);
}

// ---------------------------------------------------------------------------
// Prefill tests
// ---------------------------------------------------------------------------

#[test]
fn test_prefill_basic() {
    let num_layers = 2;
    let vocab_size = 10;
    let model = make_dummy_model(num_layers, vocab_size, vec![3]);
    let cache = KvCache::new(num_layers);
    let mut ctx = DecodeContext::new(cache, 128);

    let logits = prefill(&model, &[1, 2, 3], &mut ctx, &Device::Cpu).unwrap();
    assert_eq!(logits.dims(), &[1, vocab_size]);
    // Cache should now have seq_len=3 (the prompt length).
    assert_eq!(ctx.seq_len(), 3);
    assert_eq!(ctx.generated_count(), 0);
}

#[test]
fn test_prefill_empty_prompt_fails() {
    let model = make_dummy_model(1, 10, vec![]);
    let cache = KvCache::new(1);
    let mut ctx = DecodeContext::new(cache, 128);

    let result = prefill(&model, &[], &mut ctx, &Device::Cpu);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("empty"), "error should mention empty: {msg}");
}

#[test]
fn test_prefill_exceeds_max_seq_len() {
    let model = make_dummy_model(1, 10, vec![]);
    let cache = KvCache::new(1);
    let mut ctx = DecodeContext::new(cache, 4);

    // Prompt length 5 > max_seq_len 4.
    let result = prefill(&model, &[1, 2, 3, 4, 5], &mut ctx, &Device::Cpu);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("max_seq_len"),
        "error should mention max_seq_len: {msg}"
    );
}

#[test]
fn test_prefill_resets_state() {
    let num_layers = 2;
    let vocab_size = 10;
    let model = make_dummy_model(num_layers, vocab_size, vec![0, 0, 0, 0]);
    let cache = KvCache::new(num_layers);
    let mut ctx = DecodeContext::new(cache, 128);

    // First prefill.
    prefill(&model, &[1, 2], &mut ctx, &Device::Cpu).unwrap();
    assert_eq!(ctx.seq_len(), 2);

    // Second prefill should reset.
    prefill(&model, &[10, 20, 30], &mut ctx, &Device::Cpu).unwrap();
    assert_eq!(ctx.seq_len(), 3);
    assert_eq!(ctx.generated_count(), 0);
}

// ---------------------------------------------------------------------------
// Decode step tests
// ---------------------------------------------------------------------------

#[test]
fn test_decode_step_basic() {
    let num_layers = 2;
    let vocab_size = 10;
    let model = make_dummy_model(num_layers, vocab_size, vec![5, 7]);
    let cache = KvCache::new(num_layers);
    let mut ctx = DecodeContext::new(cache, 128);

    // Prefill with 3 tokens.
    prefill(&model, &[1, 2, 3], &mut ctx, &Device::Cpu).unwrap();
    assert_eq!(ctx.seq_len(), 3);

    // Decode one token.
    let logits = decode_step(&model, 5, &mut ctx, &Device::Cpu).unwrap();
    assert_eq!(logits.dims(), &[1, vocab_size]);
    assert_eq!(ctx.seq_len(), 4);
    assert_eq!(ctx.generated_count(), 1);
}

#[test]
fn test_decode_step_context_full() {
    let num_layers = 1;
    let vocab_size = 5;
    let model = make_dummy_model(num_layers, vocab_size, vec![0; 10]);
    let cache = KvCache::new(num_layers);
    let mut ctx = DecodeContext::new(cache, 4);

    // Prefill with 3 tokens.
    prefill(&model, &[1, 2, 3], &mut ctx, &Device::Cpu).unwrap();

    // First decode step: seq_len goes to 4 (= max_seq_len).
    decode_step(&model, 0, &mut ctx, &Device::Cpu).unwrap();
    assert!(ctx.is_full());

    // Second decode step should fail: context full.
    let result = decode_step(&model, 0, &mut ctx, &Device::Cpu);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("full"), "error should mention full: {msg}");
}

// ---------------------------------------------------------------------------
// decode_generate tests
// ---------------------------------------------------------------------------

#[test]
fn test_decode_generate_greedy() {
    let num_layers = 2;
    let vocab_size = 10;
    // Prefill returns logits with max at 3, then decode steps: 5, 7, 2 (EOS).
    let model = make_dummy_model(num_layers, vocab_size, vec![3, 5, 7, 2]);
    let cache = KvCache::new(num_layers);
    let mut ctx = DecodeContext::new(cache, 128);

    let config = GenerationConfig::new(10).with_eos_token_id(2);

    let output = decode_generate(model, &[1, 2, 3], &mut ctx, &config, &Device::Cpu).unwrap();
    // Expected: [3] from prefill, [5, 7, 2] from decode, stops at EOS=2.
    assert_eq!(output.token_ids, vec![3, 5, 7, 2]);
    assert!(output.finished);
}

#[test]
fn test_decode_generate_max_tokens() {
    let num_layers = 1;
    let vocab_size = 10;
    // All steps return token 3 (never EOS).
    let model = make_dummy_model(num_layers, vocab_size, vec![3; 20]);
    let cache = KvCache::new(num_layers);
    let mut ctx = DecodeContext::new(cache, 128);

    let config = GenerationConfig::new(5);

    let output = decode_generate(model, &[1], &mut ctx, &config, &Device::Cpu).unwrap();
    assert_eq!(output.token_ids.len(), 5);
    assert!(!output.finished);
    // All tokens should be 3.
    assert!(output.token_ids.iter().all(|&t| t == 3));
}

#[test]
fn test_decode_generate_zero_max_tokens() {
    let model = make_dummy_model(1, 10, vec![]);
    let cache = KvCache::new(1);
    let mut ctx = DecodeContext::new(cache, 128);

    let config = GenerationConfig::new(0);
    let output = decode_generate(model, &[1], &mut ctx, &config, &Device::Cpu).unwrap();
    assert!(output.token_ids.is_empty());
    assert!(!output.finished);
}

#[test]
fn test_decode_generate_empty_prompt_fails() {
    let model = make_dummy_model(1, 10, vec![]);
    let cache = KvCache::new(1);
    let mut ctx = DecodeContext::new(cache, 128);

    let config = GenerationConfig::new(10);
    let result = decode_generate(model, &[], &mut ctx, &config, &Device::Cpu);
    assert!(result.is_err());
}

#[test]
fn test_decode_generate_stops_at_context_limit() {
    let num_layers = 1;
    let vocab_size = 10;
    // All steps return token 5 (never EOS).
    let model = make_dummy_model(num_layers, vocab_size, vec![5; 100]);
    let cache = KvCache::new(num_layers);
    let mut ctx = DecodeContext::new(cache, 6); // prompt(3) + max 3 decode

    let config = GenerationConfig::new(100); // wants 100 but limited by context

    let output = decode_generate(model, &[1, 2, 3], &mut ctx, &config, &Device::Cpu).unwrap();
    // After prefill: seq=3. The is_full() check fires before each decode_step,
    // so we get: first token from prefill (no cache growth), then decode_step
    // at seq=3->4, 4->5, 5->6 (full after step), then loop checks is_full()
    // and stops. Total: 1 (prefill) + 3 (decode) = 4 tokens.
    assert_eq!(output.token_ids.len(), 4);
    assert!(!output.finished);
}

#[test]
fn test_decode_generate_eos_at_first_token() {
    let num_layers = 1;
    let vocab_size = 10;
    // Prefill returns EOS token (2) immediately.
    let model = make_dummy_model(num_layers, vocab_size, vec![2]);
    let cache = KvCache::new(num_layers);
    let mut ctx = DecodeContext::new(cache, 128);

    let config = GenerationConfig::new(10).with_eos_token_id(2);

    let output = decode_generate(model, &[1], &mut ctx, &config, &Device::Cpu).unwrap();
    assert_eq!(output.token_ids, vec![2]);
    assert!(output.finished);
}

// ---------------------------------------------------------------------------
// DecodeContext with PreallocKvCache
// ---------------------------------------------------------------------------

#[test]
fn test_decode_context_with_prealloc_cache() {
    use crate::layers::generation::PreallocKvCache;

    let cache = PreallocKvCache::new(2, 64).unwrap();
    let ctx = DecodeContext::new(cache, 64);
    assert_eq!(ctx.num_layers(), 2);
    assert_eq!(ctx.max_seq_len(), 64);
    assert_eq!(ctx.remaining_capacity(), 64);
}

// ---------------------------------------------------------------------------
// Sequence position tracking
// ---------------------------------------------------------------------------

#[test]
fn test_decode_step_increments_generated_count() {
    let num_layers = 1;
    let vocab_size = 10;
    let model = make_dummy_model(num_layers, vocab_size, vec![0; 20]);
    let cache = KvCache::new(num_layers);
    let mut ctx = DecodeContext::new(cache, 128);

    prefill(&model, &[1, 2, 3], &mut ctx, &Device::Cpu).unwrap();
    assert_eq!(ctx.generated_count(), 0);

    for i in 0..5 {
        decode_step(&model, 0, &mut ctx, &Device::Cpu).unwrap();
        assert_eq!(ctx.generated_count(), i + 1);
    }
    assert_eq!(ctx.seq_len(), 8); // 3 prompt + 5 decode
}
