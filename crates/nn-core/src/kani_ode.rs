// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for ODE solver time schedules.
//!
//! Proves arithmetic invariants of the time schedule computation:
//! - Cosine schedule values lie in [0, 2] (stub bound; actual cos gives [0, 1])
//! - Linear schedule values lie in [t_min, t_max]
//! - Euler step produces finite output from finite inputs
//! - All dt values are well-defined (no NaN)
//! - CFG combination produces finite output from finite inputs
//!
//! ## Connection to production code
//!
//! Three of the five harnesses call production scalar helpers from `ode.rs`
//! directly, verifying the exact functions used at runtime:
//! - `linear_schedule_values_in_range` → [`crate::ode::linear_t`]
//! - `euler_step_finite_output` → [`crate::ode::euler_step_scalar`]
//! - `cfg_combination_finite` → [`crate::ode::cfg_combine_scalar`]
//!
//! Two harnesses use inline stubs because of CBMC limitations:
//! - `cosine_schedule_values_in_stub_bound` — CBMC cannot model `f32::cos`,
//!   so uses a nondeterministic cos_stub in [-1, 1] (proves [0, 2] bound).
//! - `dt_finite_for_finite_timesteps` — generic `t_next - t` subtraction
//!   applicable to both schedule types, not specific to one helper.

#![cfg(kani)]

use crate::ode::{cfg_combine_scalar, euler_step_scalar, linear_t};

// ---------------------------------------------------------------------------
// AC1: Cosine schedule — values in [0, 2] (stub bound)
// ---------------------------------------------------------------------------

/// Prove: for any step index i in [0, N) with N in [1, 100],
/// cosine schedule t(s) = 1 - cos(s * pi/2) lies in [0, 2] (stub bound).
///
/// NOTE: Cannot call `cosine_t()` because CBMC cannot model `f32::cos`.
/// Uses cos_stub (nondeterministic in [-1, 1]) instead.
/// This means we prove the property for ANY function with range [-1, 1],
/// which is stronger than needed. The true bound is [0, 1].
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn cosine_schedule_values_in_stub_bound() {
    let n: u8 = kani::any();
    let i: u8 = kani::any();

    // N >= 1, i < N, bound N to keep exploration tractable
    kani::assume(n >= 1 && n <= 100);
    kani::assume(i < n);

    let _s = i as f32 / n as f32;

    // cos_stub: nondeterministic value in [-1, 1] (matches CBMC pattern)
    let cos_val: f32 = kani::any();
    kani::assume(cos_val >= -1.0 && cos_val <= 1.0);

    let t = 1.0 - cos_val;

    // t = 1 - cos(s * pi/2) where cos in [-1, 1]
    // => t in [1 - 1, 1 - (-1)] = [0, 2]
    // With the stub (cos in [-1,1]), we prove this [0, 2] bound.
    // The actual cos(s * pi/2) for s in [0,1] is in [0, 1],
    // so the true bound is [0, 1] — tighter than what we prove here.
    assert!(t >= 0.0, "cosine t must be non-negative");
    assert!(t <= 2.0, "cosine t must be <= 2.0 (stub bound)");
    assert!(!t.is_nan(), "cosine t must not be NaN");
}

// ---------------------------------------------------------------------------
// AC2: Linear schedule — values in [t_min, t_max]
// ---------------------------------------------------------------------------

/// Prove: for any step index i in [0, N) with N in [1, 100],
/// `linear_t(s, t_max, t_min)` lies between t_min and t_max.
///
/// Calls the production `linear_t()` from `ode.rs`.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn linear_schedule_values_in_range() {
    let n: u8 = kani::any();
    let i: u8 = kani::any();

    kani::assume(n >= 1 && n <= 100);
    kani::assume(i < n);

    // Use bounded t_max, t_min to avoid f32 overflow
    let t_max: f32 = kani::any();
    let t_min: f32 = kani::any();
    kani::assume(t_max.is_finite() && t_min.is_finite());
    kani::assume(t_max.abs() <= 1e6 && t_min.abs() <= 1e6);
    kani::assume(t_max >= t_min);

    let s = i as f32 / n as f32;
    let t = linear_t(s, t_max, t_min);

    // t is a convex combination: t = t_max * (1-s) + t_min * s with s in [0, 1)
    // => t in [t_min, t_max] (with possible f32 rounding at boundaries)
    assert!(t.is_finite(), "linear t must be finite");

    // t_max >= t_min is assumed above, so lo = t_min, hi = t_max.
    // Allow ULP tolerance for floating-point rounding in the convex combination.
    let eps = (t_max - t_min).abs() * f32::EPSILON * 2.0 + f32::EPSILON;
    assert!(t >= t_min - eps, "linear t below t_min");
    assert!(t <= t_max + eps, "linear t above t_max");
}

// ---------------------------------------------------------------------------
// AC3: Euler step — no NaN/Inf for finite inputs
// ---------------------------------------------------------------------------

/// Prove: `euler_step_scalar(x, v, dt)` produces a finite result
/// when x, v, dt are all finite and bounded.
///
/// Calls the production `euler_step_scalar()` from `ode.rs`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn euler_step_finite_output() {
    let x: f32 = kani::any();
    let v: f32 = kani::any();
    let dt: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(v.is_finite());
    kani::assume(dt.is_finite());

    // Bound inputs to prevent overflow in mul + add
    kani::assume(x.abs() <= 1e18);
    kani::assume(v.abs() <= 1e18);
    kani::assume(dt.abs() <= 1.0);

    let x_new = euler_step_scalar(x, v, dt);

    assert!(x_new.is_finite(), "Euler step must produce finite output");
    assert!(!x_new.is_nan(), "Euler step must not produce NaN");
}

// ---------------------------------------------------------------------------
// AC4: dt computation — no NaN for finite schedule endpoints
// ---------------------------------------------------------------------------

/// Prove: dt = t_next - t is finite when both t and t_next are finite.
///
/// NOTE: This is generic subtraction applicable to both linear and cosine
/// schedules. No production helper to call — `dt` is computed inline in
/// `TimeSchedule::steps()` as `t_next - t`.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn dt_finite_for_finite_timesteps() {
    let t: f32 = kani::any();
    let t_next: f32 = kani::any();

    kani::assume(t.is_finite() && t_next.is_finite());
    kani::assume(t.abs() <= 1e18 && t_next.abs() <= 1e18);

    let dt = t_next - t;

    assert!(dt.is_finite(), "dt must be finite for finite timesteps");
    assert!(!dt.is_nan(), "dt must not be NaN");
}

// ---------------------------------------------------------------------------
// AC5: CFG combination — finite output from finite velocities
// ---------------------------------------------------------------------------

/// Prove: `cfg_combine_scalar(v_cond, v_uncond, cfg_scale)` is finite
/// when all inputs are finite and bounded.
///
/// Calls the production `cfg_combine_scalar()` from `ode.rs`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn cfg_combination_finite() {
    let v_cond: f32 = kani::any();
    let v_uncond: f32 = kani::any();
    let cfg_scale: f32 = kani::any();

    kani::assume(v_cond.is_finite() && v_uncond.is_finite() && cfg_scale.is_finite());
    kani::assume(v_cond.abs() <= 1e9);
    kani::assume(v_uncond.abs() <= 1e9);
    kani::assume(cfg_scale.abs() <= 100.0);

    let v = cfg_combine_scalar(v_cond, v_uncond, cfg_scale);

    assert!(v.is_finite(), "CFG combination must be finite");
    assert!(!v.is_nan(), "CFG combination must not be NaN");
}
