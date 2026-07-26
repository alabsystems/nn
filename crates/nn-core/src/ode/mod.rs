// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ODE solvers for flow matching / rectified flow inference.
//!
//! Flow matching models (CosyVoice2, E2TTS, Irodori) denoise latents from
//! Gaussian noise to data by solving an ODE: `dx/dt = v(x, t)` where `v` is
//! a learned velocity field (typically a DiT network).
//!
//! This module provides:
//! - [`TimeSchedule`] — timestep schedule generation (linear, cosine)
//! - [`VelocityField`] — trait for velocity network evaluation
//! - [`euler_solve`] — Forward Euler ODE integration
//! - [`euler_solve_cfg`] — Euler integration with classifier-free guidance

use crate::error::Result;
use crate::DynTensor;

// -- Scalar helpers (Kani-verifiable, no DynTensor dependency) ----------------

/// Cosine schedule timestep: `t(s) = 1 - cos(s * pi/2)`.
///
/// Extracted so `kani_ode.rs` can verify this function directly.
/// Note: Kani uses a cos_stub (nondeterministic in \[-1, 1\]) because CBMC
/// cannot model `f32::cos`, so the Kani harness proves the weaker bound
/// `[0, 2]` rather than the true `[0, 1]`.
#[inline]
pub fn cosine_t(s: f32) -> f32 {
    1.0 - (s * std::f32::consts::FRAC_PI_2).cos()
}

/// Linear schedule timestep: `t(s) = t_max * (1 - s) + t_min * s`.
///
/// Extracted so `kani_ode.rs` can call this directly to verify the
/// convex-combination bound `[t_min, t_max]`.
#[inline]
pub fn linear_t(s: f32, t_max: f32, t_min: f32) -> f32 {
    t_max * (1.0 - s) + t_min * s
}

/// Scalar Euler step: `x_new = x + v * dt`.
///
/// The scalar equivalent of `x = x.add(&v.mul_scalar(dt))` in [`euler_solve`].
/// Extracted so Kani can verify finite-output for bounded inputs.
#[inline]
pub fn euler_step_scalar(x: f32, v: f32, dt: f32) -> f32 {
    x + v * dt
}

/// Scalar CFG velocity combination: `v = v_cond + cfg * (v_cond - v_uncond)`.
///
/// The scalar equivalent of the guidance computation in [`euler_solve_cfg`].
/// Extracted so Kani can verify finite-output for bounded inputs.
#[inline]
pub fn cfg_combine_scalar(v_cond: f32, v_uncond: f32, cfg_scale: f32) -> f32 {
    v_cond + cfg_scale * (v_cond - v_uncond)
}

/// Time schedule for ODE integration.
///
/// Controls how timesteps are distributed across integration steps.
/// Different schedules concentrate computation at different parts of
/// the trajectory.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TimeSchedule {
    /// Linear interpolation from `t_max` to `t_min` over N steps.
    ///
    /// Used by Irodori RF-DiT and Ming-omni DiTAR.
    /// Direction: `t_max` → `t_min` (typically 0.999 → 0.001, noise → clean).
    Linear {
        /// Starting time (typically close to 1.0).
        t_max: f32,
        /// Ending time (typically close to 0.0).
        t_min: f32,
    },

    /// Cosine schedule: `t(s) = 1 - cos(s * π/2)`, direction 0→1.
    ///
    /// Used by CosyVoice3 DiT. Produces non-uniform steps that are denser
    /// near t=0 (noise end), where the velocity field changes most rapidly.
    Cosine,
}

impl TimeSchedule {
    /// Generate `(t, dt)` pairs for `n` integration steps.
    ///
    /// Returns a vector of `(t, dt)` where `t` is the current timestep and
    /// `dt` is the step size to advance by. The Euler update is `x += v(x,t) * dt`.
    ///
    /// # Errors
    ///
    /// Returns an error if `n` is zero.
    pub fn steps(&self, n: usize) -> Result<Vec<(f32, f32)>> {
        if n == 0 {
            return Err(crate::TensorError::Unsupported(
                "ODE solver requires at least 1 step".into(),
            ));
        }

        match self {
            Self::Linear { t_max, t_min } => {
                if !t_max.is_finite() || !t_min.is_finite() {
                    return Err(crate::TensorError::InvalidBounds(format!(
                        "ODE linear schedule requires finite endpoints: t_max={t_max}, t_min={t_min}"
                    )));
                }
                let mut pairs = Vec::with_capacity(n);
                let n_f = n as f32;
                for i in 0..n {
                    let s = i as f32 / n_f;
                    let s_next = (i + 1) as f32 / n_f;
                    let t = linear_t(s, *t_max, *t_min);
                    let t_next = linear_t(s_next, *t_max, *t_min);
                    let dt = t_next - t;
                    pairs.push((t, dt));
                }
                Ok(pairs)
            }
            Self::Cosine => {
                let mut pairs = Vec::with_capacity(n);
                let n_f = n as f32;
                for i in 0..n {
                    let s = i as f32 / n_f;
                    let s_next = (i + 1) as f32 / n_f;
                    let t = cosine_t(s);
                    let t_next = cosine_t(s_next);
                    let dt = t_next - t;
                    pairs.push((t, dt));
                }
                Ok(pairs)
            }
        }
    }
}

/// Velocity field trait — implemented by each model's DiT/flow network.
///
/// The velocity field predicts the instantaneous rate of change of the ODE
/// state at a given timestep. For flow matching, `v(x, t)` represents the
/// learned vector field that transports noise to data.
pub trait VelocityField {
    /// Predict velocity at state `x` and time `t`.
    ///
    /// The returned tensor must have the same shape as `x`.
    fn predict(&mut self, x: &DynTensor, t: f32) -> Result<DynTensor>;
}

/// Forward Euler ODE solver for rectified flow.
///
/// Integrates `dx/dt = v(x, t)` using the Forward Euler method:
/// `x_{n+1} = x_n + v(x_n, t_n) * dt_n`
///
/// # Arguments
///
/// * `x0` — Initial state (typically Gaussian noise)
/// * `velocity` — Velocity network implementing [`VelocityField`]
/// * `schedule` — Time schedule controlling step distribution
/// * `n_steps` — Number of integration steps
///
/// # Example
///
/// ```ignore
/// // NOTE: ignore — requires a model implementing VelocityField trait
/// let noise = DynTensor::randn(0.0, 1.0, &[1, 128, 80], &Device::Cpu)?;
/// let result = euler_solve(&noise, &mut model, &TimeSchedule::Cosine, 10)?;
/// ```
pub fn euler_solve<V: VelocityField>(
    x0: &DynTensor,
    velocity: &mut V,
    schedule: &TimeSchedule,
    n_steps: usize,
) -> Result<DynTensor> {
    let pairs = schedule.steps(n_steps)?;
    let mut x = x0.clone();
    for (t, dt) in pairs {
        let v = velocity.predict(&x, t)?;
        x = x.add(&v.mul_scalar(f64::from(dt))?)?;
    }
    Ok(x)
}

/// Euler ODE solver with classifier-free guidance (CFG).
///
/// At each step, runs both conditioned and unconditioned velocity predictions
/// and combines them: `v = v_cond + cfg_scale * (v_cond - v_uncond)`.
///
/// This is the standard single-guidance CFG used by CosyVoice3 (cfg_scale=0.7).
///
/// # Arguments
///
/// * `x0` — Initial state (typically Gaussian noise)
/// * `cond_velocity` — Conditioned velocity network
/// * `uncond_velocity` — Unconditioned velocity network
/// * `schedule` — Time schedule controlling step distribution
/// * `n_steps` — Number of integration steps
/// * `cfg_scale` — Classifier-free guidance strength (0.0 = no guidance)
pub fn euler_solve_cfg<C: VelocityField, U: VelocityField>(
    x0: &DynTensor,
    cond_velocity: &mut C,
    uncond_velocity: &mut U,
    schedule: &TimeSchedule,
    n_steps: usize,
    cfg_scale: f32,
) -> Result<DynTensor> {
    if !cfg_scale.is_finite() {
        return Err(crate::TensorError::InvalidBounds(format!(
            "ODE CFG scale must be finite, got {cfg_scale}"
        )));
    }
    let pairs = schedule.steps(n_steps)?;
    let mut x = x0.clone();
    for (t, dt) in pairs {
        let v_cond = cond_velocity.predict(&x, t)?;
        let v_uncond = uncond_velocity.predict(&x, t)?;
        // v = v_cond + cfg_scale * (v_cond - v_uncond)
        let guidance = v_cond.sub(&v_uncond)?.mul_scalar(f64::from(cfg_scale))?;
        let v = v_cond.add(&guidance)?;
        x = x.add(&v.mul_scalar(f64::from(dt))?)?;
    }
    Ok(x)
}

#[cfg(test)]
mod tests;
