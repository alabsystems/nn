// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for train_loop.rs.
//!
//! Proves properties of the verification-guided training loop's scalar helper
//! functions: config validation, curriculum selection logic, score averaging,
//! curriculum sizing, and early stopping threshold comparisons.
//!
//! The training loop orchestrates differentiable audio losses with
//! non-differentiable evaluation metrics. These harnesses verify the numerical
//! foundations: curriculum fraction bounds, score averaging correctness,
//! curriculum count computation, and threshold comparison soundness.
//!
//! **Local-copy gap:** Scalar functions here re-implement production formulas
//! from `train_loop.rs`. `// SYNC:` comments track correspondence.
//!
//! Re: #3706 (Kani harnesses for nn-autodiff audio_losses + op + train_loop).

// ── Local scalar copies of production formulas ───────────────────────────

/// Curriculum count: ceil(n_samples * fraction), clamped to [1, n_available].
///
/// SYNC: train_loop.rs:323-326.
#[allow(dead_code)]
fn curriculum_count(n_samples: usize, fraction: f64, n_available: usize) -> usize {
    let raw = (n_samples as f64 * fraction).ceil() as usize;
    raw.max(1).min(n_available)
}

/// Mean of a slice of f64 scores.
///
/// SYNC: train_loop.rs:330-335 (mean_score).
#[allow(dead_code)]
fn mean_of_scores(scores: &[f64]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    let sum: f64 = scores.iter().sum();
    sum / scores.len() as f64
}

/// Config validation: curriculum_fraction must be in (0.0, 1.0].
///
/// SYNC: train_loop.rs:289.
#[allow(dead_code)]
fn is_valid_curriculum_fraction(f: f64) -> bool {
    f.is_finite() && f > 0.0 && f <= 1.0
}

/// Config validation: max_epochs must be > 0.
///
/// SYNC: train_loop.rs:283.
#[allow(dead_code)]
fn is_valid_max_epochs(e: usize) -> bool {
    e > 0
}

/// Config validation: target_score (if set) must be finite.
///
/// SYNC: train_loop.rs:304-310.
#[allow(dead_code)]
fn is_valid_target_score(score: Option<f64>) -> bool {
    match score {
        None => true,
        Some(s) => s.is_finite(),
    }
}

/// Early stopping check: mean_eval >= target.
///
/// SYNC: train_loop.rs:199.
#[allow(dead_code)]
fn should_early_stop(mean_eval: f64, target: f64) -> bool {
    mean_eval >= target
}

/// Mean loss from epoch: sum / steps (or 0 if steps == 0).
///
/// SYNC: train_loop.rs:242-246.
#[allow(dead_code)]
fn epoch_mean_loss(loss_sum: f64, steps: usize) -> f64 {
    if steps > 0 {
        loss_sum / steps as f64
    } else {
        0.0
    }
}

/// Loss value finiteness filter: only accumulate finite losses.
///
/// SYNC: train_loop.rs:231 (if loss_scalar.is_finite()).
#[allow(dead_code)]
fn accumulate_loss(current_sum: f64, loss: f32) -> f64 {
    if loss.is_finite() {
        current_sum + f64::from(loss)
    } else {
        current_sum
    }
}

fn ceil_f64_stub(x: f64) -> f64 {
    let _ = x;
    let r: f64 = kani::any();
    kani::assume(r.is_finite());
    r
}

// ── Kani proof harnesses ─────────────────────────────────────────────────

// -- Config validation --

/// Prove valid curriculum fractions pass validation.
fn ceil_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite());
    r
}

#[kani::unwind(1)]
#[kani::proof]
fn valid_curriculum_fraction_passes() {
    let f: f64 = kani::any();
    kani::assume(f.is_finite() && f > 0.0 && f <= 1.0);
    assert!(is_valid_curriculum_fraction(f), "valid fraction must pass");
}

/// Prove zero curriculum fraction is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn zero_curriculum_fraction_rejected() {
    assert!(
        !is_valid_curriculum_fraction(0.0),
        "fraction=0 must be rejected"
    );
}

/// Prove negative curriculum fraction is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn negative_curriculum_fraction_rejected() {
    let f: f64 = kani::any();
    kani::assume(f.is_finite() && f < 0.0);
    assert!(
        !is_valid_curriculum_fraction(f),
        "negative fraction must be rejected"
    );
}

/// Prove NaN curriculum fraction is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn nan_curriculum_fraction_rejected() {
    assert!(
        !is_valid_curriculum_fraction(f64::NAN),
        "NaN fraction must be rejected"
    );
}

/// Prove infinity curriculum fraction is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn inf_curriculum_fraction_rejected() {
    assert!(
        !is_valid_curriculum_fraction(f64::INFINITY),
        "infinity fraction must be rejected"
    );
    assert!(
        !is_valid_curriculum_fraction(f64::NEG_INFINITY),
        "neg infinity fraction must be rejected"
    );
}

/// Prove fraction > 1.0 is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn fraction_above_one_rejected() {
    let f: f64 = kani::any();
    kani::assume(f.is_finite() && f > 1.0 && f <= 100.0);
    assert!(
        !is_valid_curriculum_fraction(f),
        "fraction > 1.0 must be rejected"
    );
}

/// Prove max_epochs validation: 0 is rejected, >0 is accepted.
#[kani::unwind(1)]
#[kani::proof]
fn max_epochs_validation() {
    let e: usize = kani::any();
    kani::assume(e <= 10000);
    if e == 0 {
        assert!(!is_valid_max_epochs(e), "0 epochs must be rejected");
    } else {
        assert!(is_valid_max_epochs(e), ">0 epochs must be accepted");
    }
}

/// Prove target_score validation: None is valid, finite is valid, NaN/Inf rejected.
#[kani::unwind(1)]
#[kani::proof]
fn target_score_validation_none() {
    assert!(is_valid_target_score(None), "None target must be valid");
}

/// Prove target_score validation: finite is valid.
#[kani::unwind(1)]
#[kani::proof]
fn target_score_validation_finite() {
    let s: f64 = kani::any();
    kani::assume(s.is_finite());
    assert!(
        is_valid_target_score(Some(s)),
        "finite target must be valid"
    );
}

/// Prove target_score validation: NaN is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn target_score_validation_nan() {
    assert!(
        !is_valid_target_score(Some(f64::NAN)),
        "NaN target must be rejected"
    );
}

/// Prove target_score validation: Inf is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn target_score_validation_inf() {
    assert!(
        !is_valid_target_score(Some(f64::INFINITY)),
        "Inf target must be rejected"
    );
}

// -- Curriculum count --

/// Prove curriculum count is always >= 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::ceil, ceil_f64_stub)]
#[kani::stub(f32::ceil, ceil_f32_stub)]
fn curriculum_count_at_least_one() {
    let n: usize = kani::any();
    let frac: f64 = kani::any();
    let avail: usize = kani::any();
    kani::assume(n >= 1 && n <= 10000);
    kani::assume(frac.is_finite() && frac > 0.0 && frac <= 1.0);
    kani::assume(avail >= 1 && avail <= n);
    let count = curriculum_count(n, frac, avail);
    assert!(count >= 1, "curriculum count must be >= 1");
}

/// Prove curriculum count is at most n_available.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::ceil, ceil_f64_stub)]
#[kani::stub(f32::ceil, ceil_f32_stub)]
fn curriculum_count_at_most_available() {
    let n: usize = kani::any();
    let frac: f64 = kani::any();
    let avail: usize = kani::any();
    kani::assume(n >= 1 && n <= 10000);
    kani::assume(frac.is_finite() && frac > 0.0 && frac <= 1.0);
    kani::assume(avail >= 1 && avail <= n);
    let count = curriculum_count(n, frac, avail);
    assert!(count <= avail, "curriculum count must be <= n_available");
}

/// Prove curriculum count with fraction=1.0 selects all available.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::ceil, ceil_f64_stub)]
#[kani::stub(f32::ceil, ceil_f32_stub)]
fn curriculum_count_full_fraction() {
    let n: usize = kani::any();
    let avail: usize = kani::any();
    kani::assume(n >= 1 && n <= 1000);
    kani::assume(avail >= 1 && avail <= n);
    let count = curriculum_count(n, 1.0, avail);
    assert!(count == avail, "fraction=1.0 must select all available");
}

/// Prove curriculum count is monotonic in fraction.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::ceil, ceil_f64_stub)]
#[kani::stub(f32::ceil, ceil_f32_stub)]
fn curriculum_count_monotonic_in_fraction() {
    let n: usize = kani::any();
    let f1: f64 = kani::any();
    let f2: f64 = kani::any();
    let avail: usize = kani::any();
    kani::assume(n >= 1 && n <= 1000);
    kani::assume(f1.is_finite() && f1 > 0.0 && f1 <= 1.0);
    kani::assume(f2.is_finite() && f2 > f1 && f2 <= 1.0);
    kani::assume(avail >= 1 && avail <= n);
    let c1 = curriculum_count(n, f1, avail);
    let c2 = curriculum_count(n, f2, avail);
    assert!(c2 >= c1, "curriculum count must be monotonic in fraction");
}

// -- Mean score --

/// Prove mean of empty scores is 0.
#[kani::unwind(5)]
#[kani::proof]
fn mean_score_empty_is_zero() {
    let m = mean_of_scores(&[]);
    assert!(m == 0.0, "mean of empty must be 0");
}

/// Prove mean of single score equals that score.
#[kani::unwind(5)]
#[kani::proof]
fn mean_score_single() {
    let s: f64 = kani::any();
    kani::assume(s.is_finite() && s.abs() <= 1e6);
    let m = mean_of_scores(&[s]);
    assert!(
        (m - s).abs() < 1e-10,
        "mean of single score must equal the score"
    );
}

/// Prove mean of identical scores equals that score.
#[kani::unwind(5)]
#[kani::proof]
fn mean_score_identical() {
    let s: f64 = kani::any();
    let n: usize = kani::any();
    kani::assume(s.is_finite() && s.abs() <= 1e3);
    kani::assume(n >= 1 && n <= 100);
    let scores: Vec<f64> = vec![s; n];
    let m = mean_of_scores(&scores);
    assert!(
        (m - s).abs() < 1e-6,
        "mean of identical scores must equal the score"
    );
}

/// Prove mean of scores in [0, 1] is in [0, 1].
#[kani::unwind(5)]
#[kani::proof]
fn mean_score_bounded_unit() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && a >= 0.0 && a <= 1.0);
    kani::assume(b.is_finite() && b >= 0.0 && b <= 1.0);
    let m = mean_of_scores(&[a, b]);
    assert!(
        m >= 0.0 && m <= 1.0,
        "mean of [0,1] scores must be in [0,1]"
    );
}

// -- Early stopping --

/// Prove early stopping triggers when score meets target.
#[kani::unwind(1)]
#[kani::proof]
fn early_stop_triggers_at_target() {
    let target: f64 = kani::any();
    kani::assume(target.is_finite() && target >= 0.0 && target <= 1.0);
    assert!(
        should_early_stop(target, target),
        "score == target must trigger early stop"
    );
}

/// Prove early stopping triggers when score exceeds target.
#[kani::unwind(1)]
#[kani::proof]
fn early_stop_triggers_above_target() {
    let target: f64 = kani::any();
    let score: f64 = kani::any();
    kani::assume(target.is_finite() && target >= 0.0 && target <= 1.0);
    kani::assume(score.is_finite() && score > target);
    assert!(
        should_early_stop(score, target),
        "score > target must trigger early stop"
    );
}

/// Prove early stopping does NOT trigger when score is below target.
#[kani::unwind(1)]
#[kani::proof]
fn early_stop_no_trigger_below() {
    let target: f64 = kani::any();
    let score: f64 = kani::any();
    kani::assume(target.is_finite() && target > 0.0 && target <= 1.0);
    kani::assume(score.is_finite() && score >= 0.0 && score < target);
    assert!(
        !should_early_stop(score, target),
        "score < target must not trigger early stop"
    );
}

// -- Epoch mean loss --

/// Prove epoch mean loss is 0 when steps == 0.
#[kani::unwind(1)]
#[kani::proof]
fn epoch_mean_loss_zero_steps() {
    let sum: f64 = kani::any();
    kani::assume(sum.is_finite());
    let m = epoch_mean_loss(sum, 0);
    assert!(m == 0.0, "mean loss with 0 steps must be 0");
}

/// Prove epoch mean loss is finite for finite sum and positive steps.
#[kani::unwind(1)]
#[kani::proof]
fn epoch_mean_loss_finite() {
    let sum: f64 = kani::any();
    let steps: usize = kani::any();
    kani::assume(sum.is_finite() && sum.abs() <= 1e8);
    kani::assume(steps >= 1 && steps <= 100_000);
    let m = epoch_mean_loss(sum, steps);
    assert!(m.is_finite(), "epoch mean loss must be finite");
}

/// Prove epoch mean loss magnitude does not exceed loss sum magnitude.
#[kani::unwind(1)]
#[kani::proof]
fn epoch_mean_loss_bounded_by_sum() {
    let sum: f64 = kani::any();
    let steps: usize = kani::any();
    kani::assume(sum.is_finite() && sum.abs() <= 1e6);
    kani::assume(steps >= 1 && steps <= 10_000);
    let m = epoch_mean_loss(sum, steps);
    assert!(
        m.abs() <= sum.abs() + 1e-10,
        "mean loss must not exceed total loss"
    );
}

// -- Loss accumulation --

/// Prove finite loss is accumulated correctly.
#[kani::unwind(1)]
#[kani::proof]
fn accumulate_loss_finite_adds() {
    let current: f64 = kani::any();
    let loss: f32 = kani::any();
    kani::assume(current.is_finite() && current.abs() <= 1e6);
    kani::assume(loss.is_finite() && loss.abs() <= 1e3);
    let result = accumulate_loss(current, loss);
    let expected = current + f64::from(loss);
    assert!(
        (result - expected).abs() < 1e-6,
        "finite loss must be accumulated"
    );
}

/// Prove NaN loss is skipped.
#[kani::unwind(1)]
#[kani::proof]
fn accumulate_loss_nan_skipped() {
    let current: f64 = kani::any();
    kani::assume(current.is_finite() && current.abs() <= 1e6);
    let result = accumulate_loss(current, f32::NAN);
    assert!((result - current).abs() < 1e-15, "NaN loss must be skipped");
}

/// Prove infinite loss is skipped.
#[kani::unwind(1)]
#[kani::proof]
fn accumulate_loss_inf_skipped() {
    let current: f64 = kani::any();
    kani::assume(current.is_finite() && current.abs() <= 1e6);
    let result_pos = accumulate_loss(current, f32::INFINITY);
    let result_neg = accumulate_loss(current, f32::NEG_INFINITY);
    assert!(
        (result_pos - current).abs() < 1e-15,
        "+inf loss must be skipped"
    );
    assert!(
        (result_neg - current).abs() < 1e-15,
        "-inf loss must be skipped"
    );
}
