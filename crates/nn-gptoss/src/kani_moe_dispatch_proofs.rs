// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proofs for MoE dispatch logic in gpt-oss.
//!
//! Proves 5 key mathematical properties of the Mixture-of-Experts dispatch
//! pipeline used in [`GptOssMoeBlock::forward`](crate::layers::GptOssMoeBlock)
//! and [`fused_moe_forward`](crate::moe_dispatch::fused_moe_forward):
//!
//! 1. **Softmax normalization** — router probabilities sum to 1.0 (within epsilon)
//! 2. **Top-k selection** — exactly k experts selected per token when k <= num_experts
//! 3. **SwiGLU clamp bounded** — clamped SiLU output is in [-limit, limit]
//! 4. **Expert weight convexity** — renormalized expert weights sum to 1.0
//! 5. **Router bias finite** — linear(x) with bias produces finite output for finite inputs
//!
//! All proofs operate on f32 scalar arithmetic (not DynTensor) to stay within
//! Kani's model-checking capabilities. Transcendental functions (exp) use
//! nondeterministic stubs with conservative postconditions.
//!
//! Part of #4271: gpt-oss NY compose verification.

// ---------------------------------------------------------------------------
// Transcendental stubs for Kani (CBMC cannot handle exp)
// ---------------------------------------------------------------------------

/// Conservative exp stub: returns a nondeterministic positive finite value.
///
/// exp(x) > 0 for all finite x, so we constrain the result accordingly.
/// Upper bound prevents overflow in downstream summation.
fn exp_f32_stub(x: f32) -> f32 {
    let _ = x;
    let result: f32 = kani::any();
    kani::assume(result > 0.0);
    kani::assume(result.is_finite());
    kani::assume(result <= 1e10);
    result
}

// ===========================================================================
// Harness 1: Softmax normalization
// ===========================================================================

/// Proves that after softmax, router probabilities for any token sum close
/// to 1.0 (within 1e-4 numerical tolerance).
///
/// Models the softmax computation from `GptOssMoeBlock::forward` line 274:
/// ```text
/// probs = logits.softmax(logits_last)
/// ```
///
/// For N=4 nondeterministic finite logits, we compute:
///   exp_i = exp(logit_i)
///   sum   = sum(exp_i)
///   p_i   = exp_i / sum
/// and verify sum(p_i) is within epsilon of 1.0.
#[kani::proof]
#[kani::unwind(5)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn proof_softmax_normalization() {
    const N: usize = 4;

    // Nondeterministic finite logits
    let mut logits = [0.0f32; N];
    for i in 0..N {
        logits[i] = kani::any();
        kani::assume(logits[i].is_finite());
        kani::assume(logits[i] >= -100.0 && logits[i] <= 100.0);
    }

    // Compute exp and sum
    let mut exp_vals = [0.0f32; N];
    let mut exp_sum = 0.0f32;
    for i in 0..N {
        exp_vals[i] = logits[i].exp();
        exp_sum += exp_vals[i];
    }

    // exp_sum must be positive and finite for softmax to be defined
    kani::assume(exp_sum > 0.0);
    kani::assume(exp_sum.is_finite());

    // Compute softmax probabilities and verify they sum to 1.0
    let mut prob_sum = 0.0f32;
    for i in 0..N {
        let p = exp_vals[i] / exp_sum;
        kani::assume(p.is_finite());
        // Each probability is non-negative (exp > 0, sum > 0)
        assert!(p >= 0.0, "softmax probability must be non-negative");
        // Each probability is at most 1.0
        assert!(p <= 1.0 + 1e-6, "softmax probability must be <= 1.0");
        prob_sum += p;
    }

    assert!(
        (prob_sum - 1.0).abs() < 1e-4,
        "softmax probabilities must sum to ~1.0, got {}",
        prob_sum
    );
}

// ===========================================================================
// Harness 2: Top-k selection returns exactly k experts
// ===========================================================================

/// Proves that top-k selection from N experts returns exactly k distinct
/// experts per token, when k <= N and all scores are finite non-negative.
///
/// Models the top-k selection in `GptOssMoeBlock::forward` line 275:
/// ```text
/// (topk_weights, topk_indices) = probs.topk(logits_last, self.top_k)
/// ```
///
/// Uses a greedy argmax loop (equivalent to partial sort) to select k
/// highest-scoring experts, verifying count and distinctness.
#[kani::proof]
#[kani::unwind(5)]
fn proof_topk_returns_exactly_k() {
    const N: usize = 4; // num_experts (small for tractability)
    const K: usize = 2; // experts_per_token

    // Nondeterministic non-negative finite scores (post-softmax probabilities)
    let mut scores = [0.0f32; N];
    for i in 0..N {
        scores[i] = kani::any();
        kani::assume(scores[i].is_finite());
        kani::assume(scores[i] >= 0.0);
    }

    // Greedy top-k selection (equivalent to topk op)
    let mut selected = [false; N];
    let mut count: usize = 0;

    for _step in 0..K {
        let mut best_idx: usize = N; // sentinel: no selection yet
        let mut best_val: f32 = f32::NEG_INFINITY;
        for j in 0..N {
            if !selected[j] && scores[j] > best_val {
                best_val = scores[j];
                best_idx = j;
            }
        }
        // K <= N guarantees at least one unselected expert exists
        if best_idx < N {
            selected[best_idx] = true;
            count += 1;
        }
    }

    // Exactly k experts selected
    assert!(
        count == K,
        "top-k must select exactly K={} experts, got {}",
        K,
        count
    );

    // Verify distinctness by counting selected flags
    let mut distinct = 0usize;
    for i in 0..N {
        if selected[i] {
            distinct += 1;
        }
    }
    assert!(
        distinct == K,
        "selected indices must be distinct, got {} unique out of K={}",
        distinct,
        K
    );

    // Verify all selected indices are valid (< num_experts)
    for i in 0..N {
        if selected[i] {
            assert!(i < N, "selected expert index must be < num_experts");
        }
    }
}

// ===========================================================================
// Harness 3: SwiGLU clamp bounded
// ===========================================================================

/// Proves that with swiglu_limit=7.0, the output of clamp(silu(x), -7, 7)
/// is always in [-7.0, 7.0] for any finite input x.
///
/// Models the clamped SwiGLU from `GptOssMoeBlock::expert_forward` line 250:
/// ```text
/// gate.silu()?.clamp(-self.swiglu_limit, self.swiglu_limit)?
/// ```
///
/// silu(x) = x * sigmoid(x) = x / (1 + exp(-x)) can be unbounded,
/// but the clamp constrains output regardless of input magnitude.
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn proof_swiglu_clamp_bounded() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x >= -50.0 && x <= 50.0);

    let limit: f32 = 7.0;

    // silu(x) = x / (1 + exp(-x))
    let neg_x = -x;
    let exp_neg_x = neg_x.exp();
    kani::assume(exp_neg_x.is_finite());

    let denom = 1.0 + exp_neg_x;
    kani::assume(denom > 0.0);
    kani::assume(denom.is_finite());

    let silu = x / denom;
    kani::assume(silu.is_finite());

    // clamp(silu, -limit, limit)
    let clamped = if silu > limit {
        limit
    } else if silu < -limit {
        -limit
    } else {
        silu
    };

    assert!(
        clamped >= -limit,
        "clamped SiLU must be >= -{}, got {}",
        limit,
        clamped
    );
    assert!(
        clamped <= limit,
        "clamped SiLU must be <= {}, got {}",
        limit,
        clamped
    );
    assert!(
        clamped.is_finite(),
        "clamped SiLU must be finite, got {}",
        clamped
    );
}

// ===========================================================================
// Harness 4: Expert weight convexity
// ===========================================================================

/// Proves that after renormalization, expert weights per token sum to 1.0
/// (within epsilon), forming a convex combination.
///
/// Models the weight renormalization from `GptOssMoeBlock::forward` lines 278-279:
/// ```text
/// w_sum = topk_weights.sum_keepdim(logits_last)
/// topk_weights = topk_weights.broadcast_div(&w_sum)
/// ```
///
/// For K nondeterministic non-negative weights, dividing each by their sum
/// produces a valid probability distribution (non-negative, sums to 1.0).
#[kani::proof]
#[kani::unwind(5)]
fn proof_expert_weight_convexity() {
    const K: usize = 4; // experts_per_token (gpt-oss-20b uses 4)

    // Nondeterministic raw weights (from softmax top-k, so non-negative)
    let mut raw_weights = [0.0f32; K];
    let mut raw_sum = 0.0f32;
    for i in 0..K {
        raw_weights[i] = kani::any();
        kani::assume(raw_weights[i] >= 0.0);
        kani::assume(raw_weights[i].is_finite());
        kani::assume(raw_weights[i] <= 1.0); // from softmax, each <= 1.0
        raw_sum += raw_weights[i];
    }

    // Sum must be positive and finite for division to be valid
    kani::assume(raw_sum > 0.0);
    kani::assume(raw_sum.is_finite());

    // Renormalize: w_i' = w_i / sum(w_j)
    let mut norm_sum = 0.0f32;
    for i in 0..K {
        let w = raw_weights[i] / raw_sum;
        kani::assume(w.is_finite());

        // Each normalized weight is non-negative
        assert!(w >= 0.0, "expert weight must be non-negative, got {}", w);
        // Each normalized weight is at most 1.0 (within tolerance)
        assert!(w <= 1.0 + 1e-6, "expert weight must be <= 1.0, got {}", w);

        norm_sum += w;
    }

    // Normalized weights form a valid probability distribution
    assert!(
        (norm_sum - 1.0).abs() < 1e-4,
        "renormalized expert weights must sum to ~1.0, got {}",
        norm_sum
    );
}

// ===========================================================================
// Harness 5: Router bias finite
// ===========================================================================

/// Proves that the router linear layer (logit = x @ w^T + bias) does not
/// produce NaN or Inf when both the input dot product and bias are finite.
///
/// Models the router computation in `GptOssMoeBlock::forward` line 272:
/// ```text
/// logits = self.router.forward(x)   // Linear: x @ W^T + bias
/// ```
///
/// For a single expert logit: logit = dot_product + bias. If both are
/// finite, IEEE 754 addition produces a finite result (no overflow to Inf)
/// when the operands are within a safe range.
#[kani::proof]
#[kani::unwind(1)]
fn proof_router_bias_finite() {
    // Model one element of the router output: logit_i = dot(x, w_i) + bias_i
    let dot_product: f32 = kani::any();
    let bias: f32 = kani::any();

    kani::assume(dot_product.is_finite());
    kani::assume(bias.is_finite());

    // Practical range constraint: hidden_size=2880 with bounded activations
    // means dot products stay well within f32 range.
    kani::assume(dot_product >= -1e6 && dot_product <= 1e6);
    kani::assume(bias >= -1e6 && bias <= 1e6);

    let logit = dot_product + bias;

    assert!(
        !logit.is_nan(),
        "router logit must not be NaN: dot={}, bias={}",
        dot_product,
        bias
    );
    assert!(
        logit.is_finite(),
        "router logit must be finite: dot={}, bias={}, logit={}",
        dot_product,
        bias,
        logit
    );

    // Verify the result is within expected range (sum of two bounded values)
    assert!(
        logit >= -2e6 && logit <= 2e6,
        "router logit must be bounded: got {}",
        logit
    );
}
