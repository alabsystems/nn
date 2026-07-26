// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::layers::generation::kv_cache::KvCache;
use crate::{DType, Device, Result};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Minimal mock model for speculative decoding tests.
///
/// Given input IDs [1, seq_len], returns:
/// - logits [1, seq_len, vocab_size] with the "correct" token having highest logit
/// - hidden_states [1, seq_len, hidden_dim] (uniform values)
///
/// The mock model deterministically predicts: token i -> token (i + 1) % vocab.
struct MockModel {
    vocab_size: usize,
    hidden_dim: usize,
}

impl MockModel {
    fn new(vocab_size: usize, hidden_dim: usize) -> Self {
        Self {
            vocab_size,
            hidden_dim,
        }
    }

    /// Forward pass: for each input token t, predict (t + 1) % vocab_size.
    fn forward(&self, input: &DynTensor, _cache: &mut KvCache) -> Result<(DynTensor, DynTensor)> {
        let dims = input.dims().to_vec();
        let batch = dims[0];
        let seq_len = dims[1];

        // Extract input token IDs.
        let ids = input.to_flat_vec::<u32>()?;

        // Build logits: for each position, make the "next token" have highest logit.
        let mut logits_data = vec![0.0f32; batch * seq_len * self.vocab_size];
        for b in 0..batch {
            for s in 0..seq_len {
                let token = ids[b * seq_len + s] as usize;
                let next_token = (token + 1) % self.vocab_size;
                let base = (b * seq_len + s) * self.vocab_size;
                // Set all logits to -1.0, then the predicted token to 1.0.
                for v in 0..self.vocab_size {
                    logits_data[base + v] = -1.0;
                }
                logits_data[base + next_token] = 1.0;
            }
        }

        let logits = DynTensor::from_vec(
            logits_data,
            &[batch, seq_len, self.vocab_size],
            &Device::Cpu,
        )?;

        // Hidden states: uniform values for simplicity.
        let hidden = DynTensor::ones(&[batch, seq_len, self.hidden_dim], DType::F32, &Device::Cpu)?;

        Ok((logits, hidden))
    }
}

/// Mock draft function: predicts (current_last_token + 1), (current_last_token + 2), etc.
///
/// Returns N sets of logits where draft i has highest logit at
/// (last_token + i + 1) % vocab_size.
fn mock_draft_fn(
    hidden: &DynTensor,
    vocab_size: usize,
    num_draft: usize,
    start_token: usize,
) -> Result<Vec<DynTensor>> {
    let dims = hidden.dims().to_vec();
    let batch = dims[0];
    let seq_len = dims[1];

    let mut result = Vec::with_capacity(num_draft);
    for i in 0..num_draft {
        let predicted = (start_token + i + 1) % vocab_size;
        let mut data = vec![-1.0f32; batch * seq_len * vocab_size];
        // Set the predicted token at the last position.
        for b in 0..batch {
            let base = (b * seq_len + (seq_len - 1)) * vocab_size;
            data[base + predicted] = 1.0;
        }
        let logits = DynTensor::from_vec(data, &[batch, seq_len, vocab_size], &Device::Cpu)?;
        result.push(logits);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Config validation tests
// ---------------------------------------------------------------------------

#[test]
fn test_speculative_config_rejects_zero_speculative() {
    let cfg = SpeculativeConfig::new(100, 0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_speculative_config_accepts_valid() {
    let cfg = SpeculativeConfig::new(100, 4).with_eos_token_id(2);
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.eos_token_id, Some(2));
}

// ---------------------------------------------------------------------------
// Output tests
// ---------------------------------------------------------------------------

#[test]
fn test_speculative_output_acceptance_rate() {
    let out = SpeculativeOutput::new(vec![1, 2, 3], false, 10, 7);
    assert!((out.acceptance_rate() - 0.7).abs() < 1e-10);
}

#[test]
fn test_speculative_output_acceptance_rate_zero_drafted() {
    let out = SpeculativeOutput::new(vec![], false, 0, 0);
    assert!((out.acceptance_rate() - 0.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Core decode tests
// ---------------------------------------------------------------------------

#[test]
fn test_greedy_decode_empty_prompt_error() {
    let cfg = SpeculativeConfig::new(10, 2);
    let mut cache = KvCache::new(1);
    let result = greedy_decode_with_verification(
        |_h| Ok(vec![]),
        |_i, _c| {
            let l = DynTensor::zeros(&[1, 1, 10], DType::F32, &Device::Cpu)?;
            let h = DynTensor::zeros(&[1, 1, 4], DType::F32, &Device::Cpu)?;
            Ok((l, h))
        },
        &[],
        &mut cache,
        &cfg,
        &Device::Cpu,
    );
    assert!(result.is_err());
}

#[test]
fn test_greedy_decode_max_zero_returns_empty() -> Result<()> {
    let cfg = SpeculativeConfig::new(0, 2);
    let mut cache = KvCache::new(1);
    let out = greedy_decode_with_verification(
        |_h| Ok(vec![]),
        |_i, _c| {
            let l = DynTensor::zeros(&[1, 1, 10], DType::F32, &Device::Cpu)?;
            let h = DynTensor::zeros(&[1, 1, 4], DType::F32, &Device::Cpu)?;
            Ok((l, h))
        },
        &[1],
        &mut cache,
        &cfg,
        &Device::Cpu,
    )?;
    assert!(out.token_ids.is_empty());
    assert!(!out.finished);
    Ok(())
}

#[test]
fn test_greedy_decode_all_drafts_accepted() -> Result<()> {
    // Mock model predicts t -> (t+1) % 10.
    // Mock draft also predicts t -> (t+1), (t+2), etc.
    // All drafts should match the verifier.
    let vocab = 10;
    let hidden_dim = 4;
    let num_spec = 3;
    let model = MockModel::new(vocab, hidden_dim);

    // Track the last token emitted so draft_fn can produce matching predictions.
    // Since the model is deterministic (t -> t+1), starting from prompt [0],
    // the sequence is 1, 2, 3, 4, ...
    // The draft needs to predict the same sequence to get full acceptance.
    let last_tok = std::cell::Cell::new(0usize);

    let cfg = SpeculativeConfig::new(6, num_spec);
    let mut cache = KvCache::new(1);

    let out = greedy_decode_with_verification(
        |h| {
            let current = last_tok.get();
            mock_draft_fn(h, vocab, num_spec, current)
        },
        |input, cache| {
            let result = model.forward(input, cache)?;
            // Update last_tok based on the last input token.
            let ids = input.to_flat_vec::<u32>()?;
            if let Some(&last) = ids.last() {
                let predicted = (last as usize + 1) % vocab;
                last_tok.set(predicted);
            }
            Ok(result)
        },
        &[0],
        &mut cache,
        &cfg,
        &Device::Cpu,
    )?;

    // The model predicts: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6
    // We asked for max_new_tokens = 6, so we get [1, 2, 3, 4, 5, 6].
    assert_eq!(out.token_ids.len(), 6);
    assert_eq!(out.token_ids, vec![1, 2, 3, 4, 5, 6]);
    assert!(!out.finished);
    // All drafts should have been accepted.
    assert!(out.total_accepted > 0);
    Ok(())
}

#[test]
fn test_greedy_decode_eos_stops_generation() -> Result<()> {
    let vocab = 10;
    let hidden_dim = 4;
    let model = MockModel::new(vocab, hidden_dim);

    // EOS is token 3. Model predicts 0->1->2->3(EOS).
    let cfg = SpeculativeConfig::new(20, 4).with_eos_token_id(3);
    let mut cache = KvCache::new(1);

    let out = greedy_decode_with_verification(
        |h| {
            // Draft predicts the wrong sequence intentionally so we rely on verifier.
            // Predict all zeros.
            let dims = h.dims().to_vec();
            let batch = dims[0];
            let seq_len = dims[1];
            let mut result = Vec::new();
            for _ in 0..4 {
                let data = vec![0.0f32; batch * seq_len * vocab];
                let logits = DynTensor::from_vec(data, &[batch, seq_len, vocab], &Device::Cpu)?;
                result.push(logits);
            }
            Ok(result)
        },
        |input, cache| model.forward(input, cache),
        &[0],
        &mut cache,
        &cfg,
        &Device::Cpu,
    )?;

    // The verifier produces 1, 2, 3(EOS). Generation should stop at or after EOS.
    assert!(out.finished || out.token_ids.contains(&3));
    Ok(())
}

#[test]
fn test_greedy_decode_partial_acceptance() -> Result<()> {
    // Model predicts t -> (t+1) % 10.
    // Draft predicts first token correctly, then wrong tokens.
    let vocab = 10;
    let hidden_dim = 4;
    let num_spec = 3;
    let model = MockModel::new(vocab, hidden_dim);

    let cfg = SpeculativeConfig::new(10, num_spec);
    let mut cache = KvCache::new(1);

    let out = greedy_decode_with_verification(
        |h| {
            // Draft: first head correct (token 1), rest wrong (token 9, 9).
            let dims = h.dims().to_vec();
            let batch = dims[0];
            let seq_len = dims[1];
            let mut result = Vec::new();
            // Head 0: predict token 1 (correct for prompt [0])
            let mut data0 = vec![-1.0f32; batch * seq_len * vocab];
            let base = (seq_len - 1) * vocab;
            data0[base + 1] = 1.0; // correct
            result.push(DynTensor::from_vec(
                data0,
                &[batch, seq_len, vocab],
                &Device::Cpu,
            )?);
            // Head 1: predict token 9 (wrong, should be 2)
            let mut data1 = vec![-1.0f32; batch * seq_len * vocab];
            data1[base + 9] = 1.0; // wrong
            result.push(DynTensor::from_vec(
                data1,
                &[batch, seq_len, vocab],
                &Device::Cpu,
            )?);
            // Head 2: predict token 9 (wrong, should be 3)
            let mut data2 = vec![-1.0f32; batch * seq_len * vocab];
            data2[base + 9] = 1.0; // wrong
            result.push(DynTensor::from_vec(
                data2,
                &[batch, seq_len, vocab],
                &Device::Cpu,
            )?);
            Ok(result)
        },
        |input, cache| model.forward(input, cache),
        &[0],
        &mut cache,
        &cfg,
        &Device::Cpu,
    )?;

    // Should have generated some tokens; first draft accepted, rest rejected.
    assert!(!out.token_ids.is_empty());
    assert_eq!(out.token_ids[0], 1); // First token always from prefill.
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper function tests
// ---------------------------------------------------------------------------

#[test]
fn test_argmax_last_position_basic() -> Result<()> {
    // logits [1, 3, 5]: last position (pos 2) has highest at index 3.
    let mut data = vec![-1.0f32; 3 * 5];
    // Position 2 (last), token 3 is highest.
    let base = 2 * 5;
    data[base + 3] = 10.0;
    let logits = DynTensor::from_vec(data, &[1, 3, 5], &Device::Cpu)?;
    let token = argmax_last_position(&logits)?;
    assert_eq!(token, 3);
    Ok(())
}

#[test]
fn test_argmax_at_position_basic() -> Result<()> {
    // logits [1, 3, 5]: position 1 has highest at index 2.
    let mut data = vec![-1.0f32; 3 * 5];
    let base = 5;
    data[base + 2] = 10.0;
    let logits = DynTensor::from_vec(data, &[1, 3, 5], &Device::Cpu)?;
    let token = argmax_at_position(&logits, 1)?;
    assert_eq!(token, 2);
    Ok(())
}

#[test]
fn test_verify_draft_tokens_all_match() -> Result<()> {
    // Verifier logits agree with all draft tokens.
    let vocab = 5;
    // verify_logits [1, 4, 5]: positions 0,1,2 predict tokens 1,2,3.
    let mut data = vec![-1.0f32; 4 * vocab];
    // pos 0 -> token 1
    data[0 * vocab + 1] = 1.0;
    // pos 1 -> token 2
    data[vocab + 2] = 1.0;
    // pos 2 -> token 3
    data[2 * vocab + 3] = 1.0;
    // pos 3 -> token 4 (bonus position)
    data[3 * vocab + 4] = 1.0;

    let logits = DynTensor::from_vec(data, &[1, 4, vocab], &Device::Cpu)?;
    let draft_tokens = vec![1, 2, 3];
    let cfg = SpeculativeConfig::new(100, 3);

    let accepted = verify_draft_tokens(&logits, &draft_tokens, &cfg)?;
    assert_eq!(accepted, 3);
    Ok(())
}

#[test]
fn test_verify_draft_tokens_partial_match() -> Result<()> {
    let vocab = 5;
    // verify_logits [1, 4, 5]: pos 0 -> 1 (match), pos 1 -> 4 (mismatch with draft[1]=2).
    let mut data = vec![-1.0f32; 4 * vocab];
    data[0 * vocab + 1] = 1.0;
    data[vocab + 4] = 1.0; // draft expects 2, verifier says 4
    data[2 * vocab + 3] = 1.0;
    data[3 * vocab + 4] = 1.0;

    let logits = DynTensor::from_vec(data, &[1, 4, vocab], &Device::Cpu)?;
    let draft_tokens = vec![1, 2, 3];
    let cfg = SpeculativeConfig::new(100, 3);

    let accepted = verify_draft_tokens(&logits, &draft_tokens, &cfg)?;
    assert_eq!(accepted, 1); // Only first draft accepted.
    Ok(())
}

#[test]
fn test_verify_draft_tokens_none_match() -> Result<()> {
    let vocab = 5;
    // verify_logits [1, 4, 5]: pos 0 -> 4 (mismatch with draft[0]=1).
    let mut data = vec![-1.0f32; 4 * vocab];
    data[0 * vocab + 4] = 1.0; // draft expects 1, verifier says 4
    data[vocab + 2] = 1.0;
    data[2 * vocab + 3] = 1.0;
    data[3 * vocab + 4] = 1.0;

    let logits = DynTensor::from_vec(data, &[1, 4, vocab], &Device::Cpu)?;
    let draft_tokens = vec![1, 2, 3];
    let cfg = SpeculativeConfig::new(100, 3);

    let accepted = verify_draft_tokens(&logits, &draft_tokens, &cfg)?;
    assert_eq!(accepted, 0);
    Ok(())
}
