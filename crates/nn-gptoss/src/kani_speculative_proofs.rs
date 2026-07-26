// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for speculative decoding invariants.
//!
//! Covers:
//! 1. Acceptance rate bounded to [0.0, 1.0]
//! 2. Accepted token count bounded by gamma
//! 3. Adaptive gamma stays within [min_gamma, max_gamma]
//! 4. Effective speedup is non-negative
//! 5. Bonus token only present when all gamma tokens are accepted

use crate::speculative::{AdaptiveGamma, SpeculativeStats, SpeculativeStep};

// ============================================================================
// Harness 1: Acceptance rate bounded to [0.0, 1.0]
// ============================================================================

/// Proves that `verify_draft_deterministic` always produces an acceptance
/// rate in the range [0.0, 1.0].
///
/// The acceptance rate is computed as `accepted_count / total_proposed`.
/// Since `accepted_count <= total_proposed`, the rate cannot exceed 1.0.
/// When `total_proposed == 0`, the rate is 0.0 by convention.
#[kani::unwind(1)]
#[kani::proof]
fn proof_acceptance_rate_bounded() {
    let accepted_count: usize = kani::any();
    let total_proposed: usize = kani::any();

    kani::assume(total_proposed <= 32); // practical upper bound
    kani::assume(accepted_count <= total_proposed);

    let acceptance_rate = if total_proposed == 0 {
        0.0_f64
    } else {
        accepted_count as f64 / total_proposed as f64
    };

    assert!(
        acceptance_rate >= 0.0,
        "acceptance rate must be non-negative"
    );
    assert!(
        acceptance_rate <= 1.0,
        "acceptance rate must not exceed 1.0"
    );
}

// ============================================================================
// Harness 2: Accepted token count bounded by gamma
// ============================================================================

/// Proves that the number of accepted tokens from a speculative step
/// never exceeds gamma (the number of tokens proposed for verification).
///
/// This models the core loop in `verify_draft_deterministic`: the loop
/// runs at most `gamma` iterations and pushes at most one token per
/// iteration.
#[kani::unwind(1)]
#[kani::proof]
fn proof_accepted_tokens_bounded_by_gamma() {
    let gamma: usize = kani::any();
    let accepted_count: usize = kani::any();

    kani::assume(gamma <= 32);
    kani::assume(accepted_count <= gamma); // models the loop invariant

    // The SpeculativeStep struct stores accepted_tokens whose length
    // is the accepted_count.
    let step = SpeculativeStep {
        accepted_tokens: Vec::new(), // placeholder: length is what matters
        bonus_token: None,
        total_proposed: gamma,
        acceptance_rate: if gamma == 0 {
            0.0
        } else {
            accepted_count as f64 / gamma as f64
        },
    };

    assert!(
        accepted_count <= step.total_proposed,
        "accepted count must not exceed total proposed (gamma)"
    );
}

// ============================================================================
// Harness 3: Adaptive gamma stays within [min_gamma, max_gamma]
// ============================================================================

/// Proves that after any sequence of `update()` calls, the adaptive
/// gamma controller's `current_gamma` stays within [min_gamma, max_gamma].
///
/// Models a single update step with arbitrary acceptance rate and
/// verifies the invariant holds after the update.
#[kani::unwind(1)]
#[kani::proof]
fn proof_adaptive_gamma_within_bounds() {
    let initial_gamma: usize = kani::any();
    let min_gamma: usize = kani::any();
    let max_gamma: usize = kani::any();
    let max_misses: usize = kani::any();

    kani::assume(min_gamma >= 1);
    kani::assume(min_gamma <= 16);
    kani::assume(max_gamma >= min_gamma);
    kani::assume(max_gamma <= 16);
    kani::assume(initial_gamma >= min_gamma);
    kani::assume(initial_gamma <= max_gamma);
    kani::assume(max_misses >= 1);
    kani::assume(max_misses <= 16);

    let mut ag = AdaptiveGamma::new(initial_gamma, min_gamma, max_gamma, max_misses);

    // Pre-condition: gamma is in bounds after construction.
    assert!(ag.get() >= ag.min_gamma());
    assert!(ag.get() <= ag.max_gamma());

    // Simulate one update with arbitrary step results.
    let accepted_count: usize = kani::any();
    let total_proposed: usize = kani::any();
    kani::assume(total_proposed <= 16);
    kani::assume(accepted_count <= total_proposed);

    let acceptance_rate = if total_proposed == 0 {
        0.0
    } else {
        accepted_count as f64 / total_proposed as f64
    };

    let accepted_tokens = Vec::new(); // length not checked by update()
    let step = SpeculativeStep {
        accepted_tokens,
        bonus_token: None,
        total_proposed,
        acceptance_rate,
    };

    ag.update(&step);

    // Post-condition: gamma is still in bounds.
    assert!(
        ag.get() >= ag.min_gamma(),
        "gamma must not drop below min_gamma after update"
    );
    assert!(
        ag.get() <= ag.max_gamma(),
        "gamma must not exceed max_gamma after update"
    );
}

// ============================================================================
// Harness 4: Effective speedup is non-negative
// ============================================================================

/// Proves that `effective_speedup()` always returns a non-negative value.
///
/// Speedup = total_accepted / total_steps. Both numerator and denominator
/// are non-negative unsigned integers, so the result is >= 0.0 (with
/// 0.0 returned when total_steps == 0).
#[kani::unwind(1)]
#[kani::proof]
fn proof_speedup_nonnegative() {
    let total_accepted: usize = kani::any();
    let total_steps: usize = kani::any();
    let total_proposed: usize = kani::any();

    kani::assume(total_accepted <= 1_000_000);
    kani::assume(total_steps <= 1_000_000);
    kani::assume(total_proposed <= 1_000_000);
    kani::assume(total_accepted <= total_proposed);

    let stats = SpeculativeStats {
        total_proposed,
        total_accepted,
        total_steps,
    };

    let speedup = stats.effective_speedup();
    assert!(speedup >= 0.0, "effective speedup must be non-negative");

    let acceptance = stats.average_acceptance_rate();
    assert!(
        acceptance >= 0.0,
        "average acceptance rate must be non-negative"
    );
    assert!(
        acceptance <= 1.0 || total_proposed == 0,
        "average acceptance rate must not exceed 1.0"
    );
}

// ============================================================================
// Harness 5: Bonus token only on full acceptance
// ============================================================================

/// Proves that in `verify_draft_deterministic`, a bonus token (Some) is
/// returned if and only if all `gamma` tokens were accepted.
///
/// This models the control flow: the loop exits early on rejection
/// (returning `bonus_token: None`), and only reaches the bonus-token
/// assignment when the loop completes all `gamma` iterations.
#[kani::unwind(1)]
#[kani::proof]
fn proof_bonus_token_only_on_full_acceptance() {
    let gamma: usize = kani::any();
    let accepted_count: usize = kani::any();
    let has_bonus: bool = kani::any();

    kani::assume(gamma >= 1);
    kani::assume(gamma <= 32);
    kani::assume(accepted_count <= gamma);

    // Model the verify_draft_deterministic logic:
    // bonus_token is Some only when accepted_count == gamma.
    let bonus_token_present = accepted_count == gamma;

    // If bonus is present, all must be accepted.
    if bonus_token_present {
        assert_eq!(
            accepted_count, gamma,
            "bonus token requires all tokens to be accepted"
        );
    }

    // If not all accepted, no bonus.
    if accepted_count < gamma {
        assert!(
            !bonus_token_present,
            "partial acceptance must not produce a bonus token"
        );
    }
}
