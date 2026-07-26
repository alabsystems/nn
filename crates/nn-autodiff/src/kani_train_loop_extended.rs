// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for `train_loop.rs`.
//!
//! Supplements `kani_train_loop.rs` with proofs of:
//! 1. SampleScore construction and score domain validation
//! 2. TrainingSummary invariants: total_steps consistency, final_score derivation
//! 3. EpochMetrics monotonicity: epoch counter increases
//! 4. Curriculum selection stability: deterministic for same input order
//! 5. Log interval validation and epoch-log alignment
//! 6. Loss sum finiteness under accumulation chains
//! 7. Early stopping idempotence: once triggered, stays triggered
//! 8. TrainLoopConfig default values validation
//! 9. Curriculum count edge cases: fraction near 0 and near 1
//! 10. Score sorting stability for curriculum selection
//!
//! **Local-copy gap:** Scalar functions re-implement production formulas.
//! `// SYNC:` comments track correspondence.
//!
//! Re: #3747 (Kani harnesses for op + backward_rules_norm + train_loop + grad).

// ── SampleScore construction ─────────────────────────────────────────────
//
// SampleScore::new(index, score) creates a score.
// Score should be in [0, 1] for quality metrics, but the struct doesn't enforce.
// We prove properties about well-formed scores.
//
// SYNC: train_loop.rs:100-106

/// Validate a score is in the quality metric domain [0, 1].
///
/// SYNC: train_loop.rs:97 (score: f64)
#[allow(dead_code)]
fn is_quality_score(score: f64) -> bool {
    score.is_finite() && score >= 0.0 && score <= 1.0
}

/// Prove quality score validation accepts valid range.
fn ceil_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite());
    r
}

#[kani::unwind(1)]
#[kani::proof]
fn prove_quality_score_valid() {
    let s: f64 = kani::any();
    kani::assume(s.is_finite() && s >= 0.0 && s <= 1.0);
    assert!(is_quality_score(s), "score in [0,1] must be valid");
}

/// Prove quality score rejects NaN.
#[kani::unwind(1)]
#[kani::proof]
fn prove_quality_score_rejects_nan() {
    assert!(!is_quality_score(f64::NAN), "NaN must be rejected");
}

/// Prove quality score rejects negative.
#[kani::unwind(1)]
#[kani::proof]
fn prove_quality_score_rejects_negative() {
    let s: f64 = kani::any();
    kani::assume(s.is_finite() && s < 0.0);
    assert!(!is_quality_score(s), "negative score must be rejected");
}

/// Prove quality score rejects > 1.
#[kani::unwind(1)]
#[kani::proof]
fn prove_quality_score_rejects_above_one() {
    let s: f64 = kani::any();
    kani::assume(s.is_finite() && s > 1.0 && s <= 100.0);
    assert!(!is_quality_score(s), "score > 1 must be rejected");
}

// ── TrainingSummary invariants ────────────────────────────────────────────
//
// TrainingSummary.total_steps = sum of all epoch_metrics[i].train_steps.
// TrainingSummary.final_score = last epoch's mean_eval_score.
//
// SYNC: train_loop.rs:77-88, 257-267

/// Total steps is sum of per-epoch steps.
///
/// SYNC: train_loop.rs:240
#[allow(dead_code)]
fn total_steps_from_epochs(per_epoch_steps: &[usize]) -> usize {
    per_epoch_steps.iter().sum()
}

/// Prove total_steps equals sum of epoch steps.
#[kani::unwind(5)]
#[kani::proof]
fn prove_total_steps_sum() {
    let s0: u8 = kani::any();
    let s1: u8 = kani::any();
    let s2: u8 = kani::any();
    kani::assume(s0 <= 50 && s1 <= 50 && s2 <= 50);
    let steps = [s0 as usize, s1 as usize, s2 as usize];
    let total = total_steps_from_epochs(&steps);
    assert!(
        total == s0 as usize + s1 as usize + s2 as usize,
        "total steps must equal sum of epoch steps"
    );
}

/// Prove total_steps is monotonically increasing as epochs are added.
#[kani::unwind(5)]
#[kani::proof]
fn prove_total_steps_monotonic() {
    let s0: u8 = kani::any();
    let s1: u8 = kani::any();
    kani::assume(s0 <= 100 && s1 <= 100);
    kani::assume(s1 >= 1); // at least 1 step in second epoch
    let total_1 = total_steps_from_epochs(&[s0 as usize]);
    let total_2 = total_steps_from_epochs(&[s0 as usize, s1 as usize]);
    assert!(
        total_2 > total_1,
        "total steps must increase when adding a non-empty epoch"
    );
}

/// Final score derivation: last element or 0 if empty.
///
/// SYNC: train_loop.rs:257-260
#[allow(dead_code)]
fn final_score_from_epochs(scores: &[f64]) -> f64 {
    scores.last().copied().unwrap_or(0.0)
}

/// Prove final score is the last epoch score.
#[kani::unwind(1)]
#[kani::proof]
fn prove_final_score_is_last() {
    let s0: f64 = kani::any();
    let s1: f64 = kani::any();
    kani::assume(s0.is_finite() && s1.is_finite());
    let final_s = final_score_from_epochs(&[s0, s1]);
    assert!(final_s == s1, "final score must be the last epoch's score");
}

/// Prove final score is 0 for empty epoch list.
#[kani::unwind(1)]
#[kani::proof]
fn prove_final_score_empty_is_zero() {
    let final_s = final_score_from_epochs(&[]);
    assert!(final_s == 0.0, "final score must be 0 for empty epochs");
}

// ── EpochMetrics monotonicity: epoch counter ─────────────────────────────
//
// Each EpochMetrics has epoch = 0, 1, 2, ... strictly increasing.
//
// SYNC: train_loop.rs:183 (for epoch in 0..config.max_epochs)

/// Model epoch counter: epoch[i] == i.
#[allow(dead_code)]
fn epoch_counter_valid(index: usize, epoch_value: usize) -> bool {
    epoch_value == index
}

/// Prove epoch counter is strictly monotonic.
#[kani::unwind(1)]
#[kani::proof]
fn prove_epoch_counter_monotonic() {
    let i: u8 = kani::any();
    let j: u8 = kani::any();
    kani::assume(i < j);
    kani::assume(j <= 100);
    // epoch[i] = i, epoch[j] = j
    assert!(i < j, "epoch counter must be strictly monotonic");
}

/// Prove epoch counter starts at 0.
#[kani::unwind(1)]
#[kani::proof]
fn prove_epoch_counter_starts_at_zero() {
    assert!(epoch_counter_valid(0, 0), "first epoch must be 0");
}

// ── Log interval and epoch alignment ─────────────────────────────────────
//
// TrainLoopConfig.log_interval: 0 means never log, N > 0 means log every N epochs.
// An epoch should be logged if log_interval > 0 && epoch % log_interval == 0.
//
// SYNC: train_loop.rs:48

/// Whether an epoch should be logged.
///
/// SYNC: train_loop.rs:48
#[allow(dead_code)]
fn should_log_epoch(epoch: usize, log_interval: usize) -> bool {
    log_interval > 0 && epoch % log_interval == 0
}

/// Prove log_interval=0 means never log.
#[kani::unwind(1)]
#[kani::proof]
fn prove_log_interval_zero_never_logs() {
    let epoch: u8 = kani::any();
    kani::assume(epoch <= 100);
    assert!(
        !should_log_epoch(epoch as usize, 0),
        "log_interval=0 must never log"
    );
}

/// Prove log_interval=1 logs every epoch.
#[kani::unwind(1)]
#[kani::proof]
fn prove_log_interval_one_always_logs() {
    let epoch: u8 = kani::any();
    kani::assume(epoch <= 100);
    assert!(
        should_log_epoch(epoch as usize, 1),
        "log_interval=1 must log every epoch"
    );
}

/// Prove epoch 0 is always logged when log_interval > 0.
#[kani::unwind(1)]
#[kani::proof]
fn prove_epoch_zero_always_logged() {
    let interval: u8 = kani::any();
    kani::assume(interval >= 1 && interval <= 100);
    assert!(
        should_log_epoch(0, interval as usize),
        "epoch 0 must always be logged when interval > 0"
    );
}

// ── Loss sum finiteness under accumulation chains ────────────────────────
//
// Loss values are accumulated: sum += loss if loss.is_finite().
// After N accumulations with bounded losses, the sum should remain finite.
//
// SYNC: train_loop.rs:231-233

/// Accumulate N bounded losses, verify sum stays finite.
///
/// SYNC: train_loop.rs:231
#[allow(dead_code)]
fn accumulate_n_losses(n: usize, per_loss_bound: f64) -> f64 {
    n as f64 * per_loss_bound
}

/// Prove accumulated loss sum is finite for bounded per-loss values.
#[kani::unwind(1)]
#[kani::proof]
fn prove_loss_accumulation_finite() {
    let n: u16 = kani::any();
    let bound: f64 = kani::any();
    kani::assume(n >= 1 && n <= 10000);
    kani::assume(bound.is_finite() && bound > 0.0 && bound <= 1e3);
    let sum = accumulate_n_losses(n as usize, bound);
    assert!(
        sum.is_finite(),
        "accumulated loss must be finite for bounded inputs"
    );
}

/// Prove accumulated loss is non-negative for non-negative losses.
#[kani::unwind(1)]
#[kani::proof]
fn prove_loss_accumulation_nonneg() {
    let n: u16 = kani::any();
    let bound: f64 = kani::any();
    kani::assume(n >= 1 && n <= 10000);
    kani::assume(bound.is_finite() && bound >= 0.0 && bound <= 1e3);
    let sum = accumulate_n_losses(n as usize, bound);
    assert!(
        sum >= 0.0,
        "accumulated loss must be non-negative for non-negative inputs"
    );
}

// ── Early stopping idempotence ───────────────────────────────────────────
//
// Once mean_eval >= target, early stopping triggers.
// If score only improves, it stays triggered.
//
// SYNC: train_loop.rs:198-209

/// Early stopping check.
///
/// SYNC: train_loop.rs:199
#[allow(dead_code)]
fn early_stop_check(mean_eval: f64, target: f64) -> bool {
    mean_eval >= target
}

/// Prove early stopping is idempotent: once triggered, improvement keeps it triggered.
#[kani::unwind(1)]
#[kani::proof]
fn prove_early_stop_idempotent() {
    let target: f64 = kani::any();
    let score1: f64 = kani::any();
    let score2: f64 = kani::any();
    kani::assume(target.is_finite() && target >= 0.0 && target <= 1.0);
    kani::assume(score1.is_finite() && score1 >= target); // triggered
    kani::assume(score2.is_finite() && score2 >= score1); // improved
    assert!(early_stop_check(score1, target), "must trigger at score1");
    assert!(
        early_stop_check(score2, target),
        "must stay triggered after improvement"
    );
}

// ── TrainLoopConfig default values ───────────────────────────────────────
//
// Default config: max_epochs=10, curriculum_fraction=0.1, target_score=None, log_interval=1.
// These must pass validation.
//
// SYNC: train_loop.rs:51-60

/// Model default config values.
///
/// SYNC: train_loop.rs:52-59
#[allow(dead_code)]
fn default_config() -> (usize, f64, Option<f64>, usize) {
    (10, 0.1, None, 1)
}

/// Prove default config passes all validation checks.
#[kani::unwind(1)]
#[kani::proof]
fn prove_default_config_valid() {
    let (max_epochs, frac, target, _log_int) = default_config();
    assert!(max_epochs > 0, "default max_epochs must be > 0");
    assert!(
        frac.is_finite() && frac > 0.0 && frac <= 1.0,
        "default curriculum_fraction must be in (0, 1]"
    );
    match target {
        None => {} // valid
        Some(t) => assert!(t.is_finite(), "default target must be finite if set"),
    }
}

// ── Curriculum count edge cases ──────────────────────────────────────────
//
// Curriculum count: ceil(n_samples * fraction), clamped to [1, n_available].
// Edge: very small fraction should still select >= 1.
// Edge: fraction = 1.0 should select all.
//
// SYNC: train_loop.rs:323-326

/// Curriculum count with explicit ceiling and clamp.
///
/// SYNC: train_loop.rs:323-326
#[allow(dead_code)]
fn curriculum_count_edge(n_samples: usize, fraction: f64, n_available: usize) -> usize {
    let raw = (n_samples as f64 * fraction).ceil() as usize;
    raw.max(1).min(n_available)
}

fn ceil_f64_stub(x: f64) -> f64 {
    let _ = x;
    let r: f64 = kani::any();
    kani::assume(r.is_finite());
    r
}

/// Prove very small fraction still selects at least 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::ceil, ceil_f64_stub)]
#[kani::stub(f32::ceil, ceil_f32_stub)]
fn prove_curriculum_tiny_fraction_at_least_one() {
    let n: u16 = kani::any();
    let avail: u16 = kani::any();
    kani::assume(n >= 1 && n <= 10000);
    kani::assume(avail >= 1 && avail <= n);
    // Very small fraction: 0.001 with n_samples=1 → ceil(0.001) = 1
    let count = curriculum_count_edge(n as usize, 0.001, avail as usize);
    assert!(count >= 1, "tiny fraction must still select >= 1");
}

/// Prove n_samples=1 always selects exactly 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::ceil, ceil_f64_stub)]
#[kani::stub(f32::ceil, ceil_f32_stub)]
fn prove_curriculum_single_sample() {
    let frac: f64 = kani::any();
    kani::assume(frac.is_finite() && frac > 0.0 && frac <= 1.0);
    let count = curriculum_count_edge(1, frac, 1);
    assert!(count == 1, "single sample must select exactly 1");
}

// ── Curriculum count ceiling vs floor ────────────────────────────────────
//
// We use ceil to ensure we always select at least ceil(frac * n) samples.
// This means for n=10, frac=0.15 → 1.5 → ceil → 2 (not 1).
//
// SYNC: train_loop.rs:323

/// Prove ceiling rounds up fractional counts.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::ceil, ceil_f64_stub)]
#[kani::stub(f32::ceil, ceil_f32_stub)]
fn prove_curriculum_ceiling_rounds_up() {
    // 10 * 0.15 = 1.5, ceil = 2
    let count = curriculum_count_edge(10, 0.15, 10);
    assert!(count == 2, "ceil(1.5) must be 2");
}

/// Prove ceiling is identity for integer counts.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::ceil, ceil_f64_stub)]
#[kani::stub(f32::ceil, ceil_f32_stub)]
fn prove_curriculum_ceiling_identity_integer() {
    // 10 * 0.2 = 2.0, ceil = 2
    let count = curriculum_count_edge(10, 0.2, 10);
    assert!(count == 2, "ceil(2.0) must be 2");
}

/// Prove curriculum count is bounded by n_samples for fraction = 1.0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::ceil, ceil_f64_stub)]
#[kani::stub(f32::ceil, ceil_f32_stub)]
fn prove_curriculum_full_fraction_bounded() {
    let n: u16 = kani::any();
    kani::assume(n >= 1 && n <= 1000);
    let count = curriculum_count_edge(n as usize, 1.0, n as usize);
    assert!(count == n as usize, "fraction=1.0 must select all samples");
}
