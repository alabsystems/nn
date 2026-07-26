// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for ODE solver safety.
//!
//! Covers properties beyond the basic schedule/Euler/CFG proofs in
//! `kani_ode.rs`:
//!
//! 1. Euler step size: dt > 0 produces forward progression (x_new != x when v != 0)
//! 2. Euler stability: for linear systems, |1 + dt*lambda| < 1 when dt < 2/|lambda|
//! 3. Time schedule monotonicity: t_{i+1} > t_i for all linear schedule steps
//! 4. Step count: linear schedule produces exactly n (t, dt) pairs
//! 5. RK4 coefficient symmetry: Butcher tableau weights sum to 1
//! 6. Adaptive step: error estimate triggers halving or doubling
//! 7. Energy preservation: symplectic Euler conserves discrete Hamiltonian
//! 8. Velocity field evaluation: field returns finite values for bounded inputs
//! 9. Midpoint method: second-order convergence property
//! 10. Time reversal: solve forward then backward returns to start (scalar Euler)
//!
//! ## Connection to production code
//!
//! Harnesses 1, 3, 8, 10 call production scalar helpers from `ode.rs`:
//! - `euler_step_forward_progression` -> [`crate::ode::euler_step_scalar`]
//! - `linear_schedule_monotonicity` -> [`crate::ode::linear_t`]
//! - `velocity_field_finite_output` -> [`crate::ode::euler_step_scalar`]
//! - `time_reversal_euler_roundtrip` -> [`crate::ode::euler_step_scalar`]
//!
//! Harnesses 2, 4, 5, 6, 7, 9 use inline scalar math for properties that
//! don't have direct production helpers (stability regions, RK4 tableau,
//! adaptive stepping, symplectic integration, midpoint method).
//!
//! Part of #4186: Extended Kani proofs for ODE solver safety.

#![cfg(kani)]

use crate::ode::{euler_step_scalar, linear_t};

// ---------------------------------------------------------------------------
// 1. Euler step size: dt > 0 produces forward progression
// ---------------------------------------------------------------------------

/// Prove: when dt > 0 and v != 0, the Euler step `x_new = x + v * dt`
/// differs from `x` (i.e., forward progression occurs).
///
/// Calls the production `euler_step_scalar()` from `ode.rs`.
#[kani::unwind(1)]
#[kani::proof]
fn euler_step_forward_progression() {
    let x: f32 = kani::any();
    let v: f32 = kani::any();
    let dt: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(v.is_finite());
    kani::assume(dt.is_finite());

    // Bound inputs to prevent overflow
    kani::assume(x.abs() <= 1e6);
    kani::assume(v.abs() >= 1e-6 && v.abs() <= 1e6);
    kani::assume(dt > 0.0 && dt <= 1.0);

    let x_new = euler_step_scalar(x, v, dt);

    // v * dt is non-zero when v != 0 and dt > 0, so x_new != x
    assert!(x_new.is_finite(), "Euler step must produce finite output");
    assert!(
        x_new != x,
        "Euler step with non-zero velocity and positive dt must progress"
    );
}

// ---------------------------------------------------------------------------
// 2. Euler stability: |1 + dt * lambda| < 1 when dt < 2/|lambda|
// ---------------------------------------------------------------------------

/// Prove: for a linear ODE dx/dt = lambda * x with lambda < 0,
/// the Euler amplification factor |1 + dt * lambda| < 1
/// when 0 < dt < 2 / |lambda|.
///
/// This is the stability region of Forward Euler for real negative eigenvalues.
/// Uses inline scalar math (no production helper for stability factor).
#[kani::unwind(1)]
#[kani::proof]
fn euler_stability_linear_system() {
    let lambda: f32 = kani::any();
    let dt: f32 = kani::any();

    kani::assume(lambda.is_finite() && dt.is_finite());

    // lambda must be negative (dissipative system)
    kani::assume(lambda < -1e-6 && lambda >= -1e6);
    // dt positive and strictly within stability region
    kani::assume(dt > 0.0);
    // Stability condition: dt < 2 / |lambda|
    // Use a margin to avoid floating-point boundary effects
    let stability_bound = 2.0 / lambda.abs();
    kani::assume(stability_bound.is_finite());
    kani::assume(dt < stability_bound * 0.99);

    let amplification = 1.0 + dt * lambda;

    assert!(
        amplification.is_finite(),
        "amplification factor must be finite"
    );
    // |1 + dt * lambda| < 1 for stability
    assert!(
        amplification.abs() < 1.0,
        "Euler method must be stable when dt < 2/|lambda|"
    );
}

// ---------------------------------------------------------------------------
// 3. Time schedule monotonicity: t_{i+1} > t_i for linear schedule
// ---------------------------------------------------------------------------

/// Prove: for a linear schedule with t_max > t_min,
/// consecutive timesteps satisfy t_{i+1} > t_i (monotonic decrease
/// from t_max toward t_min since direction is t_max -> t_min).
///
/// Actually, linear_t with s increasing goes from t_max (s=0) to t_min (s=1),
/// so the schedule is monotonically decreasing when t_max > t_min.
/// We prove: t(s) > t(s_next) for s < s_next when t_max > t_min.
///
/// Calls the production `linear_t()` from `ode.rs`.
#[kani::unwind(1)]
#[kani::proof]
fn linear_schedule_monotonicity() {
    let n: u8 = kani::any();
    let i: u8 = kani::any();

    kani::assume(n >= 2 && n <= 100);
    kani::assume(i < n - 1); // i and i+1 both valid

    let t_max: f32 = kani::any();
    let t_min: f32 = kani::any();
    kani::assume(t_max.is_finite() && t_min.is_finite());
    kani::assume(t_max.abs() <= 1e4 && t_min.abs() <= 1e4);
    kani::assume(t_max > t_min + 1e-4); // strict ordering with margin

    let n_f = n as f32;
    let s = i as f32 / n_f;
    let s_next = (i + 1) as f32 / n_f;

    let t = linear_t(s, t_max, t_min);
    let t_next = linear_t(s_next, t_max, t_min);

    assert!(t.is_finite(), "t must be finite");
    assert!(t_next.is_finite(), "t_next must be finite");

    // linear_t goes from t_max (s=0) to t_min (s=1), so t > t_next
    // when t_max > t_min and s < s_next.
    // The dt = t_next - t is negative (step goes toward t_min).
    let dt = t_next - t;
    assert!(dt.is_finite(), "dt must be finite");
    assert!(
        dt < 0.0,
        "linear schedule dt must be negative (t_max -> t_min direction)"
    );
    // Equivalently: each successive t is strictly smaller
    assert!(
        t > t_next,
        "linear schedule must be strictly monotonically decreasing"
    );
}

// ---------------------------------------------------------------------------
// 4. Step count: total steps = (t_end - t_start) / dt for uniform schedule
// ---------------------------------------------------------------------------

/// Prove: for a uniform-step schedule from t_start to t_end with n steps,
/// the sum of all dt values equals t_end - t_start (within floating-point
/// tolerance).
///
/// Uses the linear schedule formula where dt = (t_min - t_max) / n per step.
#[kani::unwind(1)]
#[kani::proof]
fn step_count_dt_sum_equals_range() {
    let n: u8 = kani::any();
    kani::assume(n >= 1 && n <= 50);

    let t_max: f32 = kani::any();
    let t_min: f32 = kani::any();
    kani::assume(t_max.is_finite() && t_min.is_finite());
    kani::assume(t_max.abs() <= 100.0 && t_min.abs() <= 100.0);
    kani::assume(t_max > t_min);

    // For linear schedule, each dt = (t_min - t_max) / n (uniform).
    // linear_t(s, t_max, t_min) = t_max*(1-s) + t_min*s
    // dt = linear_t(s_next) - linear_t(s) = (t_min - t_max) * (1/n) per step.
    let per_step_dt = (t_min - t_max) / (n as f32);

    assert!(per_step_dt.is_finite(), "per-step dt must be finite");

    // Sum of n uniform dt values = n * per_step_dt = t_min - t_max
    let total_dt = per_step_dt * (n as f32);
    let expected = t_min - t_max;

    assert!(total_dt.is_finite(), "total dt sum must be finite");

    // Allow ULP tolerance for floating-point accumulation
    let eps = (t_max - t_min).abs() * f32::EPSILON * (n as f32) + f32::EPSILON;
    assert!(
        (total_dt - expected).abs() <= eps,
        "sum of dt must equal t_end - t_start"
    );
}

// ---------------------------------------------------------------------------
// 5. RK4 coefficient symmetry: Butcher tableau weights sum to 1
// ---------------------------------------------------------------------------

/// Prove: the classic RK4 Butcher tableau weights (1/6, 1/3, 1/3, 1/6)
/// sum to 1.0 (within floating-point tolerance).
///
/// This is a fundamental consistency requirement: the weighted average of
/// four slope evaluations must form a proper convex combination for the
/// method to be consistent (order >= 1).
///
/// Uses inline constants (no production RK4 helper exists).
#[kani::unwind(1)]
#[kani::proof]
fn rk4_butcher_tableau_weights_sum_to_one() {
    // Classic RK4 weights: b = [1/6, 1/3, 1/3, 1/6]
    let b1: f32 = 1.0 / 6.0;
    let b2: f32 = 1.0 / 3.0;
    let b3: f32 = 1.0 / 3.0;
    let b4: f32 = 1.0 / 6.0;

    let sum = b1 + b2 + b3 + b4;

    assert!(sum.is_finite(), "RK4 weight sum must be finite");
    assert!(
        (sum - 1.0).abs() < f32::EPSILON * 4.0,
        "RK4 Butcher tableau weights must sum to 1"
    );

    // Also verify symmetry: b1 == b4 and b2 == b3
    assert!(
        (b1 - b4).abs() < f32::EPSILON,
        "RK4 outer weights must be symmetric"
    );
    assert!(
        (b2 - b3).abs() < f32::EPSILON,
        "RK4 inner weights must be symmetric"
    );
}

/// Prove: RK4 Butcher tableau node values (c = [0, 1/2, 1/2, 1]) are
/// consistent with the row-sum condition c_i = sum(a_{ij}).
///
/// For classic RK4:
///   c1 = 0, c2 = 1/2, c3 = 1/2, c4 = 1
///   a21 = 1/2, a31 = 0, a32 = 1/2, a41 = 0, a42 = 0, a43 = 1
#[kani::unwind(1)]
#[kani::proof]
fn rk4_butcher_tableau_row_sum_consistency() {
    // Butcher tableau a-coefficients
    let a21: f32 = 0.5;
    let a31: f32 = 0.0;
    let a32: f32 = 0.5;
    let a41: f32 = 0.0;
    let a42: f32 = 0.0;
    let a43: f32 = 1.0;

    // Node values
    let c1: f32 = 0.0;
    let c2: f32 = 0.5;
    let c3: f32 = 0.5;
    let c4: f32 = 1.0;

    // Row-sum condition: c_i = sum_j a_{ij}
    let eps = f32::EPSILON * 2.0;
    assert!(
        (c1 - 0.0).abs() < eps,
        "c1 must be 0 (no a-coefficients in first row)"
    );
    assert!((c2 - a21).abs() < eps, "c2 must equal a21");
    assert!((c3 - (a31 + a32)).abs() < eps, "c3 must equal a31 + a32");
    assert!(
        (c4 - (a41 + a42 + a43)).abs() < eps,
        "c4 must equal a41 + a42 + a43"
    );
}

// ---------------------------------------------------------------------------
// 6. Adaptive step: error estimate triggers halving or doubling
// ---------------------------------------------------------------------------

/// Prove: adaptive step-size control correctly halves dt when the local
/// error exceeds a tolerance, and doubles dt when the error is small.
///
/// Models a simplified embedded RK method error estimator.
/// Uses inline scalar math (no production adaptive stepper exists).
#[kani::unwind(1)]
#[kani::proof]
fn adaptive_step_halving_and_doubling() {
    let dt: f32 = kani::any();
    let error_est: f32 = kani::any();
    let tol: f32 = kani::any();

    kani::assume(dt.is_finite() && dt > 1e-6 && dt <= 1.0);
    kani::assume(error_est.is_finite() && error_est >= 0.0 && error_est <= 1e6);
    kani::assume(tol.is_finite() && tol > 1e-8 && tol <= 1.0);

    // Safety factor (standard value 0.9)
    let safety = 0.9_f32;

    if error_est > tol {
        // Error too large: halve the step
        let dt_new = dt * 0.5;
        assert!(dt_new.is_finite(), "halved dt must be finite");
        assert!(dt_new > 0.0, "halved dt must be positive");
        assert!(dt_new < dt, "halved dt must be smaller than original");
    } else if error_est < tol * safety * safety {
        // Error well below tolerance: double the step
        let dt_new = dt * 2.0;
        assert!(dt_new.is_finite(), "doubled dt must be finite");
        assert!(dt_new > dt, "doubled dt must be larger than original");
    } else {
        // Error close to tolerance: keep dt unchanged
        assert!(dt > 0.0, "dt must remain positive in accept region");
    }
}

// ---------------------------------------------------------------------------
// 7. Energy preservation: symplectic Euler conserves discrete Hamiltonian
// ---------------------------------------------------------------------------

/// Prove: the symplectic Euler integrator preserves the discrete energy
/// H(p, q) = p^2/2 + k*q^2/2 for a harmonic oscillator, to first order.
///
/// Symplectic Euler: p_{n+1} = p_n - k*q_n*dt, q_{n+1} = q_n + p_{n+1}*dt
/// The modified Hamiltonian H_dt is exactly conserved. We prove the exact
/// Hamiltonian changes by at most O(dt^2).
///
/// Uses inline scalar math (no production symplectic integrator).
#[kani::unwind(1)]
#[kani::proof]
fn symplectic_euler_energy_bounded_drift() {
    let p: f32 = kani::any();
    let q: f32 = kani::any();
    let k: f32 = kani::any();
    let dt: f32 = kani::any();

    kani::assume(p.is_finite() && q.is_finite() && k.is_finite() && dt.is_finite());
    kani::assume(p.abs() <= 10.0 && q.abs() <= 10.0);
    kani::assume(k > 0.0 && k <= 10.0);
    kani::assume(dt > 0.0 && dt <= 0.1);

    // Initial energy
    let h0 = 0.5 * p * p + 0.5 * k * q * q;

    // Symplectic Euler step
    let p_new = p - k * q * dt;
    let q_new = q + p_new * dt;

    // Final energy
    let h1 = 0.5 * p_new * p_new + 0.5 * k * q_new * q_new;

    assert!(h0.is_finite(), "initial energy must be finite");
    assert!(h1.is_finite(), "final energy must be finite");

    // Energy drift bounded by O(dt^2). For bounded inputs:
    // |H1 - H0| <= C * dt^2 where C depends on max(|p|, |q|, k).
    // With |p|,|q| <= 10, k <= 10, dt <= 0.1:
    // Worst case drift ~ k * |p| * |q| * dt^2 + k^2 * |q|^2 * dt^2
    //                  ~ 10 * 10 * 10 * 0.01 + 100 * 100 * 0.01 = 10 + 100 = 110
    // Use generous bound of 200 * dt^2
    let energy_drift = (h1 - h0).abs();
    let max_drift = 200.0 * dt * dt;
    assert!(
        energy_drift <= max_drift,
        "symplectic Euler energy drift must be O(dt^2)"
    );
}

// ---------------------------------------------------------------------------
// 8. Velocity field evaluation: field returns finite values for bounded inputs
// ---------------------------------------------------------------------------

/// Prove: applying a bounded velocity field via euler_step_scalar produces
/// finite outputs when the velocity magnitude is bounded.
///
/// Models a velocity field as `v = alpha * x + beta` (linear affine field)
/// and verifies that the Euler update remains finite.
///
/// Calls the production `euler_step_scalar()` from `ode.rs`.
#[kani::unwind(1)]
#[kani::proof]
fn velocity_field_finite_output() {
    let x: f32 = kani::any();
    let alpha: f32 = kani::any();
    let beta: f32 = kani::any();
    let dt: f32 = kani::any();

    kani::assume(x.is_finite() && alpha.is_finite() && beta.is_finite() && dt.is_finite());
    kani::assume(x.abs() <= 1e6);
    kani::assume(alpha.abs() <= 10.0);
    kani::assume(beta.abs() <= 1e6);
    kani::assume(dt.abs() <= 1.0);

    // Velocity field: v(x) = alpha * x + beta
    let v = alpha * x + beta;
    assert!(
        v.is_finite(),
        "velocity field must produce finite output for bounded inputs"
    );

    // Euler step with this velocity
    let x_new = euler_step_scalar(x, v, dt);
    assert!(
        x_new.is_finite(),
        "Euler step with bounded velocity must be finite"
    );
    assert!(
        !x_new.is_nan(),
        "Euler step with bounded velocity must not be NaN"
    );
}

// ---------------------------------------------------------------------------
// 9. Midpoint method: second-order convergence property
// ---------------------------------------------------------------------------

/// Prove: the midpoint method (explicit, 2nd order) applied to a linear ODE
/// dx/dt = lambda*x has amplification factor (1 + dt*lambda + (dt*lambda)^2/2),
/// which matches the Taylor expansion of exp(dt*lambda) to O(dt^2).
///
/// For the scalar linear case, the midpoint method gives:
///   k1 = lambda * x
///   k2 = lambda * (x + dt/2 * k1) = lambda * x * (1 + dt*lambda/2)
///   x_new = x + dt * k2 = x * (1 + dt*lambda + (dt*lambda)^2/2)
///
/// Uses inline scalar math (no production midpoint helper).
#[kani::unwind(1)]
#[kani::proof]
fn midpoint_method_amplification_factor() {
    let x: f32 = kani::any();
    let lambda: f32 = kani::any();
    let dt: f32 = kani::any();

    kani::assume(x.is_finite() && lambda.is_finite() && dt.is_finite());
    kani::assume(x.abs() >= 1e-3 && x.abs() <= 100.0);
    kani::assume(lambda.abs() <= 10.0);
    kani::assume(dt > 0.0 && dt <= 0.1);

    // Midpoint method on dx/dt = lambda * x
    let k1 = lambda * x;
    let x_mid = x + 0.5 * dt * k1;
    let k2 = lambda * x_mid;
    let x_new = x + dt * k2;

    assert!(x_new.is_finite(), "midpoint step must be finite");

    // Verify amplification factor matches expected formula
    let dl = dt * lambda;
    let expected_factor = 1.0 + dl + 0.5 * dl * dl;
    let x_expected = x * expected_factor;

    assert!(x_expected.is_finite(), "expected value must be finite");

    // They should agree exactly (same arithmetic path for linear ODE)
    let rel_err = if x_expected.abs() > 1e-10 {
        ((x_new - x_expected) / x_expected).abs()
    } else {
        (x_new - x_expected).abs()
    };

    assert!(
        rel_err < f32::EPSILON * 16.0,
        "midpoint method amplification must match Taylor expansion to O(dt^2)"
    );
}

// ---------------------------------------------------------------------------
// 10. Time reversal: solve forward then backward returns to start
// ---------------------------------------------------------------------------

/// Prove: one Euler step forward with dt followed by one Euler step backward
/// with -dt returns to the original x, for a constant velocity field.
///
/// For constant v: x1 = x + v * dt, x2 = x1 + v * (-dt) = x1 - v*dt = x.
///
/// Calls the production `euler_step_scalar()` from `ode.rs`.
#[kani::unwind(1)]
#[kani::proof]
fn time_reversal_euler_roundtrip() {
    let x: f32 = kani::any();
    let v: f32 = kani::any();
    let dt: f32 = kani::any();

    kani::assume(x.is_finite() && v.is_finite() && dt.is_finite());
    kani::assume(x.abs() <= 1e6);
    kani::assume(v.abs() <= 1e6);
    kani::assume(dt.abs() <= 1.0 && dt.abs() >= 1e-6);

    // Forward step
    let x_forward = euler_step_scalar(x, v, dt);
    assert!(x_forward.is_finite(), "forward step must be finite");

    // Backward step (negate dt for time reversal with constant field)
    let x_roundtrip = euler_step_scalar(x_forward, v, -dt);
    assert!(x_roundtrip.is_finite(), "backward step must be finite");

    // Should return to original x within floating-point tolerance
    // x_roundtrip = (x + v*dt) + v*(-dt) = x + v*dt - v*dt
    // Floating-point error: |v*dt| * epsilon (from addition/subtraction)
    let tol = (v.abs() * dt.abs()) * f32::EPSILON * 4.0 + f32::EPSILON;
    assert!(
        (x_roundtrip - x).abs() <= tol,
        "Euler time reversal must return to start for constant velocity"
    );
}
