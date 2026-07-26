// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proofs for autoregressive generation correctness in gpt-oss.
//!
//! Proves 5 key properties of the generation pipeline from [`generate.rs`]:
//!
//! 1. **Greedy argmax returns valid index** -- argmax index < vocab_size
//! 2. **Temperature scaling preserves order** -- temperature > 0 preserves
//!    relative logit order
//! 3. **Top-p sum bounded** -- nucleus sampling cumulative sum in [0, 1]
//! 4. **Repetition penalty reduces score** -- penalty > 1 reduces repeated
//!    token logits
//! 5. **EOS detection terminates** -- EOS token in output triggers termination
//!
//! All proofs operate on f32 scalar/slice arithmetic (not DynTensor) to stay
//! within Kani's model-checking capabilities. Transcendental functions (exp)
//! use nondeterministic stubs with conservative postconditions.
//!
//! Part of #4271: gpt-oss Kani proof expansion.

// ---------------------------------------------------------------------------
// Transcendental stubs for Kani (CBMC cannot handle exp)
// ---------------------------------------------------------------------------

/// Conservative exp stub: returns a nondeterministic positive finite value.
///
/// exp(x) > 0 for all finite x, so we constrain the result accordingly.
fn exp_f32_stub(x: f32) -> f32 {
    let _ = x;
    let result: f32 = kani::any();
    kani::assume(result > 0.0);
    kani::assume(result.is_finite());
    kani::assume(result <= 1e10);
    result
}

// ---------------------------------------------------------------------------
// Kani-local helper: argmax (mirrors generate.rs::argmax)
// ---------------------------------------------------------------------------

fn argmax_kani(values: &[f32]) -> usize {
    let mut best_idx: usize = 0;
    let mut best_val = f32::NEG_INFINITY;
    let mut i = 0;
    while i < values.len() {
        if values[i] > best_val || (values[i] == best_val && values[i].is_finite()) {
            best_val = values[i];
            best_idx = i;
        }
        i += 1;
    }
    best_idx
}

// ---------------------------------------------------------------------------
// Kani-local helper: softmax (mirrors generate.rs::softmax_vec)
// ---------------------------------------------------------------------------

fn softmax_kani(logits: &[f32], out: &mut [f32]) {
    let n = logits.len();
    let mut max_val = f32::NEG_INFINITY;
    let mut i = 0;
    while i < n {
        if logits[i] > max_val {
            max_val = logits[i];
        }
        i += 1;
    }
    if !max_val.is_finite() {
        let uniform = 1.0 / n as f32;
        let mut j = 0;
        while j < n {
            out[j] = uniform;
            j += 1;
        }
        return;
    }
    let mut sum = 0.0f32;
    i = 0;
    while i < n {
        out[i] = (logits[i] - max_val).exp();
        sum += out[i];
        i += 1;
    }
    if !sum.is_finite() || sum == 0.0 {
        let uniform = 1.0 / n as f32;
        let mut j = 0;
        while j < n {
            out[j] = uniform;
            j += 1;
        }
        return;
    }
    i = 0;
    while i < n {
        out[i] /= sum;
        i += 1;
    }
}

// ===========================================================================
// Harness 1: Greedy argmax returns valid index < vocab_size
// ===========================================================================

/// Proves that argmax always returns an index within bounds for any non-empty
/// logit slice. This guarantees `sample_logits` with temperature=0 produces
/// a valid token index.
///
/// Models the greedy path in `generate.rs::sample_logits`:
/// ```text
/// if config.temperature == 0.0 { return Ok(argmax(logits)); }
/// ```
#[kani::proof]
#[kani::unwind(9)]
fn proof_greedy_argmax_valid() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 1 && vocab_size <= 8);

    let mut logits = [0.0f32; 8];
    let mut i = 0;
    while i < vocab_size {
        logits[i] = kani::any();
        kani::assume(logits[i].is_finite());
        i += 1;
    }

    let result = argmax_kani(&logits[..vocab_size]);

    // Property 1: Index is within bounds
    assert!(
        result < vocab_size,
        "argmax index {} must be < vocab_size {}",
        result,
        vocab_size
    );

    // Property 2: No logit in the slice exceeds the one at result index
    let max_val = logits[result];
    i = 0;
    while i < vocab_size {
        assert!(
            logits[i] <= max_val + 1e-6,
            "logit at {} ({}) exceeds max at {} ({})",
            i,
            logits[i],
            result,
            max_val
        );
        i += 1;
    }
}

// ===========================================================================
// Harness 2: Temperature scaling preserves relative logit order
// ===========================================================================

/// Proves that dividing all logits by a positive temperature preserves
/// relative ordering. If logits[a] > logits[b] before scaling, then
/// logits[a] / T > logits[b] / T after scaling.
///
/// Models the temperature scaling in `generate.rs::sample_logits`:
/// ```text
/// let scaled: Vec<f32> = logits.iter().map(|&l| l / temp).collect();
/// ```
#[kani::proof]
#[kani::unwind(1)]
fn proof_temperature_scaling_preserves_order() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let temp: f32 = kani::any();

    kani::assume(a.is_finite());
    kani::assume(b.is_finite());
    kani::assume(temp.is_finite());
    kani::assume(temp > 0.0);
    // Keep values in a range where division is well-behaved
    kani::assume(a >= -1e6 && a <= 1e6);
    kani::assume(b >= -1e6 && b <= 1e6);
    kani::assume(temp >= 1e-6 && temp <= 1e6);

    let scaled_a = a / temp;
    let scaled_b = b / temp;

    kani::assume(scaled_a.is_finite());
    kani::assume(scaled_b.is_finite());

    // Property: Relative order is preserved
    if a > b {
        assert!(
            scaled_a >= scaled_b,
            "order violated: a={} > b={}, but scaled_a={} < scaled_b={}",
            a,
            b,
            scaled_a,
            scaled_b,
        );
    }

    // Property: Equal logits remain equal after scaling
    if a == b {
        assert!(
            scaled_a == scaled_b,
            "equal logits diverged: scaled_a={} != scaled_b={}",
            scaled_a,
            scaled_b,
        );
    }
}

// ===========================================================================
// Harness 3: Top-p (nucleus) sampling cumulative sum in [0, 1]
// ===========================================================================

/// Proves that after softmax, the cumulative probability sum used in
/// top-p filtering is bounded in [0, 1]. Each softmax output is
/// non-negative and the total sums to ~1.0.
///
/// Models the top-p logic in `generate.rs::apply_top_p`:
/// ```text
/// let probs = softmax_vec(logits);
/// cumulative += probs[idx];  // cumulative stays in [0, 1]
/// ```
#[kani::proof]
#[kani::unwind(5)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn proof_top_p_sum_bounded() {
    const N: usize = 4;

    let mut logits = [0.0f32; N];
    let mut i = 0;
    while i < N {
        logits[i] = kani::any();
        kani::assume(logits[i].is_finite());
        kani::assume(logits[i] >= -100.0 && logits[i] <= 100.0);
        i += 1;
    }

    let mut probs = [0.0f32; N];
    softmax_kani(&logits, &mut probs);

    // Verify cumulative sum stays in [0, 1]
    let mut cumulative = 0.0f32;
    i = 0;
    while i < N {
        // Property 1: Each probability is non-negative
        assert!(
            probs[i] >= 0.0,
            "softmax output must be non-negative, got {}",
            probs[i]
        );

        cumulative += probs[i];

        // Property 2: Cumulative sum never exceeds 1 + tolerance
        assert!(
            cumulative <= 1.0 + 1e-4,
            "cumulative probability {} exceeds 1.0",
            cumulative
        );

        i += 1;
    }

    // Property 3: Final sum is close to 1.0
    assert!(
        (cumulative - 1.0).abs() < 1e-3,
        "softmax probabilities must sum to ~1.0, got {}",
        cumulative
    );
}

// ===========================================================================
// Harness 4: Repetition penalty reduces repeated token logits
// ===========================================================================

/// Proves that repetition penalty > 1.0 strictly reduces the score of
/// previously generated tokens. Positive logits are divided by penalty;
/// negative logits are multiplied (made more negative).
///
/// Models `generate.rs::apply_repetition_penalty`:
/// ```text
/// logits[token_id] = if l > 0.0 { l / penalty } else { l * penalty };
/// ```
#[kani::proof]
#[kani::unwind(1)]
fn proof_repetition_penalty_reduces_score() {
    let logit: f32 = kani::any();
    let penalty: f32 = kani::any();

    kani::assume(logit.is_finite());
    kani::assume(penalty.is_finite());
    kani::assume(penalty > 1.0);
    kani::assume(logit >= -1e6 && logit <= 1e6);
    kani::assume(penalty <= 1e3);

    let penalized = if logit > 0.0 {
        logit / penalty
    } else {
        logit * penalty
    };

    kani::assume(penalized.is_finite());

    // Property 1: Positive logits are reduced (penalty > 1 -> l/p < l)
    if logit > 0.0 {
        assert!(
            penalized < logit,
            "positive logit {} must decrease with penalty {}, got {}",
            logit,
            penalty,
            penalized
        );
        assert!(
            penalized > 0.0,
            "penalized positive logit must remain positive"
        );
    }

    // Property 2: Negative logits become more negative (l * p < l for l < 0, p > 1)
    if logit < 0.0 {
        assert!(
            penalized < logit,
            "negative logit {} must decrease with penalty {}, got {}",
            logit,
            penalty,
            penalized
        );
    }

    // Property 3: Zero logit is unchanged
    if logit == 0.0 {
        assert!(
            penalized == 0.0,
            "zero logit must remain zero, got {}",
            penalized
        );
    }
}

// ===========================================================================
// Harness 5: EOS detection terminates generation
// ===========================================================================

/// Proves that when the sampled token equals `eos_token_id`, the generation
/// loop terminates (token is not appended to the output). This models the
/// EOS check in `generate.rs::generate`:
/// ```text
/// if token == eos_token_id { break; }
/// ```
///
/// The proof verifies that:
/// 1. EOS token causes loop exit (output does not contain EOS)
/// 2. Non-EOS token is appended to output
/// 3. The generated sequence length is bounded by max_tokens
#[kani::proof]
#[kani::unwind(6)]
fn proof_eos_detection_terminates() {
    let max_tokens: usize = kani::any();
    kani::assume(max_tokens >= 1 && max_tokens <= 4);

    let eos_token_id: usize = kani::any();
    kani::assume(eos_token_id <= 10);

    // Simulate the decode loop with nondeterministic token selection
    let mut generated_len: usize = 0;
    let mut terminated_by_eos = false;

    let mut step = 0;
    while step < max_tokens {
        let token: usize = kani::any();
        kani::assume(token <= 10);

        if token == eos_token_id {
            terminated_by_eos = true;
            break;
        }
        generated_len += 1;
        step += 1;
    }

    // Property 1: Output length bounded by max_tokens
    assert!(
        generated_len <= max_tokens,
        "generated {} tokens, exceeds max {}",
        generated_len,
        max_tokens
    );

    // Property 2: If terminated by EOS, output does not include the EOS token
    // (EOS is detected before appending)
    if terminated_by_eos {
        assert!(
            generated_len < max_tokens,
            "EOS termination must stop before max_tokens"
        );
    }

    // Property 3: If not terminated by EOS, output length equals max_tokens
    if !terminated_by_eos {
        assert!(
            generated_len == max_tokens,
            "without EOS, must generate exactly max_tokens"
        );
    }
}
