// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Speculative decoding using [`MtpHead`] multi-token predictions.
//!
//! Implements the speculative decoding pattern from Leviathan et al. (2023) and
//! the GLM-OCR/DeepSeek-V3 MTP architecture:
//!
//! 1. **Draft:** The MTP head predicts N future tokens greedily from the
//!    current hidden state.
//! 2. **Verify:** A single model forward pass processes all N draft tokens
//!    simultaneously.
//! 3. **Accept:** We accept the longest prefix of draft tokens that the
//!    verifier agrees with, then fall back to the verifier's prediction for
//!    the first rejected position.
//!
//! This yields 1-to-N accepted tokens per verification step, amortizing the
//! cost of the full model forward pass.
//!
//! # Usage
//!
//! ```ignore
//! let output = greedy_decode_with_verification(
//!     |hidden| mtp_head.forward_per_head(hidden),
//!     |input_ids, cache| model.forward(input_ids, cache),
//!     &prompt_ids,
//!     &mut cache,
//!     &SpeculativeConfig::new(100, 4),
//!     &Device::Cpu,
//! )?;
//! ```

use super::kv_cache::KvCacheBackend;
use crate::dyn_tensor::DynTensor;
use crate::{Device, Result, TensorError};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for speculative decoding with MTP.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,

    /// Number of draft tokens to predict per step (matches MTP head count).
    pub num_speculative: usize,

    /// Token ID that signals end of generation.
    pub eos_token_id: Option<usize>,
}

impl SpeculativeConfig {
    /// Create a speculative decoding config.
    #[must_use]
    pub fn new(max_new_tokens: usize, num_speculative: usize) -> Self {
        Self {
            max_new_tokens,
            num_speculative,
            eos_token_id: None,
        }
    }

    /// Set end-of-sequence token ID.
    #[must_use]
    pub fn with_eos_token_id(mut self, eos_token_id: usize) -> Self {
        self.eos_token_id = Some(eos_token_id);
        self
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.num_speculative == 0 {
            return Err(TensorError::InvalidShape(
                "SpeculativeConfig: num_speculative must be > 0".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// Output from speculative decoding.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SpeculativeOutput {
    /// Generated token IDs (not including the prompt).
    pub token_ids: Vec<usize>,

    /// Whether generation stopped due to EOS token (vs max_new_tokens).
    pub finished: bool,

    /// Total number of draft tokens proposed across all steps.
    pub total_drafted: usize,

    /// Total number of draft tokens accepted by the verifier.
    pub total_accepted: usize,
}

impl SpeculativeOutput {
    /// Create a new speculative output.
    pub(crate) fn new(
        token_ids: Vec<usize>,
        finished: bool,
        total_drafted: usize,
        total_accepted: usize,
    ) -> Self {
        Self {
            token_ids,
            finished,
            total_drafted,
            total_accepted,
        }
    }

    /// Acceptance rate: fraction of draft tokens accepted by verifier.
    ///
    /// Returns 0.0 if no tokens were drafted.
    #[must_use]
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_drafted == 0 {
            return 0.0;
        }
        self.total_accepted as f64 / self.total_drafted as f64
    }
}

// ---------------------------------------------------------------------------
// Core decode loop
// ---------------------------------------------------------------------------

/// Greedy speculative decoding with MTP draft + single-pass verification.
///
/// # Arguments
///
/// - `draft_fn`: Given the hidden states `[B, T, D]` from the model backbone,
///   returns per-head logits as `Vec<DynTensor>` each shaped `[B, T, V]`.
///   Typically `|h| mtp_head.forward_per_head(h)`.
///
/// - `model_fn`: Full model forward pass. Takes `(input_ids, &mut cache)`,
///   returns a tuple `(logits [B, T, V], hidden_states [B, T, D])`.
///   The logits are used for verification; the hidden states feed the draft.
///
/// - `prompt_ids`: Initial token IDs for prefill.
///
/// - `cache`: KV cache backend (mutated during generation).
///
/// - `config`: Speculative decoding parameters.
///
/// - `device`: Where to allocate token tensors.
///
/// # Returns
///
/// [`SpeculativeOutput`] with generated tokens and acceptance statistics.
pub fn greedy_decode_with_verification<C, D, M>(
    draft_fn: D,
    model_fn: M,
    prompt_ids: &[usize],
    cache: &mut C,
    config: &SpeculativeConfig,
    device: &Device,
) -> Result<SpeculativeOutput>
where
    C: KvCacheBackend,
    D: Fn(&DynTensor) -> Result<Vec<DynTensor>>,
    M: Fn(&DynTensor, &mut C) -> Result<(DynTensor, DynTensor)>,
{
    config.validate()?;

    if prompt_ids.is_empty() {
        return Err(TensorError::InvalidShape(
            "greedy_decode_with_verification: prompt_ids must not be empty".into(),
        ));
    }
    if config.max_new_tokens == 0 {
        return Ok(SpeculativeOutput::new(Vec::new(), false, 0, 0));
    }

    let mut generated = Vec::with_capacity(config.max_new_tokens);
    let mut total_drafted: usize = 0;
    let mut total_accepted: usize = 0;

    // Prefill: run the full prompt through the model.
    let prompt_tensor = ids_to_tensor(prompt_ids, device)?;
    let (logits, hidden_states) = model_fn(&prompt_tensor, cache)?;

    // Sample first token from the last position of prefill logits.
    let first_token = argmax_last_position(&logits)?;

    if is_eos(first_token, config) {
        generated.push(first_token);
        return Ok(SpeculativeOutput::new(generated, true, 0, 0));
    }
    generated.push(first_token);

    if generated.len() >= config.max_new_tokens {
        return Ok(SpeculativeOutput::new(generated, false, 0, 0));
    }

    // Use the hidden states from prefill for the first draft.
    let mut last_hidden = hidden_states;
    let mut last_token = first_token;

    // Main speculative decode loop.
    loop {
        if generated.len() >= config.max_new_tokens {
            break;
        }

        // Step 1: Draft N tokens using the MTP head.
        let draft_logits = draft_fn(&last_hidden)?;
        let num_draft = draft_logits
            .len()
            .min(config.max_new_tokens - generated.len());

        let mut draft_tokens = Vec::with_capacity(num_draft);
        for logits_i in draft_logits.iter().take(num_draft) {
            let token = argmax_last_position(logits_i)?;
            draft_tokens.push(token);
        }
        total_drafted += draft_tokens.len();

        // Step 2: Build verification input: [last_accepted_token] + draft_tokens.
        let mut verify_ids = Vec::with_capacity(1 + draft_tokens.len());
        verify_ids.push(last_token);
        verify_ids.extend_from_slice(&draft_tokens);
        let verify_tensor = ids_to_tensor(&verify_ids, device)?;

        let (verify_logits, verify_hidden) = model_fn(&verify_tensor, cache)?;

        // Step 3: Verify draft tokens.
        // verify_logits is [1, 1+N, V]. Position 0 predicts what comes after
        // last_token (should match draft_tokens[0]). Position i predicts what
        // comes after draft_tokens[i-1] (should match draft_tokens[i]).
        let accepted = verify_draft_tokens(&verify_logits, &draft_tokens, config)?;
        total_accepted += accepted;

        // Accept the verified prefix.
        let mut hit_eos = false;
        for &tok in draft_tokens.iter().take(accepted) {
            generated.push(tok);
            if is_eos(tok, config) {
                hit_eos = true;
                break;
            }
        }

        if hit_eos || generated.len() >= config.max_new_tokens {
            return Ok(SpeculativeOutput::new(
                generated,
                hit_eos,
                total_drafted,
                total_accepted,
            ));
        }

        // If not all draft tokens accepted, take the verifier's prediction
        // at the rejection point.
        if accepted < draft_tokens.len() {
            let fallback_token = argmax_at_position(&verify_logits, accepted)?;
            generated.push(fallback_token);
            last_token = fallback_token;

            if is_eos(fallback_token, config) || generated.len() >= config.max_new_tokens {
                return Ok(SpeculativeOutput::new(
                    generated,
                    is_eos(fallback_token, config),
                    total_drafted,
                    total_accepted,
                ));
            }
        } else {
            // All accepted: take the verifier's next-token prediction
            // (position after the last draft token).
            let bonus_token = argmax_at_position(&verify_logits, draft_tokens.len())?;
            generated.push(bonus_token);
            last_token = bonus_token;

            if is_eos(bonus_token, config) || generated.len() >= config.max_new_tokens {
                return Ok(SpeculativeOutput::new(
                    generated,
                    is_eos(bonus_token, config),
                    total_drafted,
                    total_accepted,
                ));
            }
        }

        // Update hidden states for next draft round.
        last_hidden = verify_hidden;
    }

    Ok(SpeculativeOutput::new(
        generated,
        false,
        total_drafted,
        total_accepted,
    ))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Verify draft tokens against verifier logits.
///
/// Returns the number of accepted tokens (longest matching prefix).
///
/// `verify_logits` shape: `[1, 1+N, V]` where position `i` contains logits
/// for predicting what comes after the i-th input token.
fn verify_draft_tokens(
    verify_logits: &DynTensor,
    draft_tokens: &[usize],
    config: &SpeculativeConfig,
) -> Result<usize> {
    let mut accepted = 0;
    for (i, &draft_tok) in draft_tokens.iter().enumerate() {
        let verifier_tok = argmax_at_position(verify_logits, i)?;
        if verifier_tok != draft_tok {
            break;
        }
        accepted += 1;
        if is_eos(draft_tok, config) {
            break;
        }
    }
    Ok(accepted)
}

/// Argmax over the last sequence position of logits `[B, T, V]`.
///
/// Returns the token ID with highest logit at position T-1.
fn argmax_last_position(logits: &DynTensor) -> Result<usize> {
    let rank = logits.rank();
    if rank < 2 {
        return Err(TensorError::RankMismatch {
            expected: 2,
            actual: rank,
        });
    }
    let seq_dim = rank - 2;
    let seq_len = logits.dim(seq_dim)?;
    if seq_len == 0 {
        return Err(TensorError::InvalidShape(
            "argmax_last_position: sequence length is 0".into(),
        ));
    }
    argmax_at_position(logits, seq_len - 1)
}

/// Argmax at a specific sequence position of logits `[B, T, V]`.
///
/// Returns the token ID with highest logit at the given position.
fn argmax_at_position(logits: &DynTensor, pos: usize) -> Result<usize> {
    let rank = logits.rank();
    if rank < 2 {
        return Err(TensorError::RankMismatch {
            expected: 2,
            actual: rank,
        });
    }
    let seq_dim = rank - 2;
    // Narrow to [B, 1, V], then squeeze to [B, V] or [V].
    let slice = logits.narrow(seq_dim, pos, 1)?;
    let squeezed = slice.squeeze(seq_dim)?;

    // Argmax over the vocab dimension (last dim).
    let vocab_dim = squeezed.rank() - 1;
    let indices = squeezed.argmax(vocab_dim)?;
    let flat = indices.to_flat_vec::<u32>()?;
    if flat.is_empty() {
        return Err(TensorError::InvalidShape(
            "argmax_at_position: empty argmax result".into(),
        ));
    }
    Ok(flat[0] as usize)
}

/// Convert token IDs to a 2D DynTensor `[1, seq_len]` with U32 dtype.
fn ids_to_tensor(ids: &[usize], device: &Device) -> Result<DynTensor> {
    let data: Vec<u32> = ids
        .iter()
        .map(|&id| {
            u32::try_from(id).map_err(|_| TensorError::ValueOutOfRange {
                description: "token id exceeds u32::MAX",
            })
        })
        .collect::<Result<Vec<_>>>()?;
    DynTensor::from_vec_u32(data, &[1, ids.len()], device)
}

/// Check if a token matches the EOS token ID.
fn is_eos(token: usize, config: &SpeculativeConfig) -> bool {
    config.eos_token_id.is_some_and(|eos| token == eos)
}

#[cfg(test)]
#[path = "mtp_speculative_tests.rs"]
mod tests;
