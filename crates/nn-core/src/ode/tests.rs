// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for ODE solvers and time schedules.
//!
//! CFG solver and NaN/Inf validation tests extracted to
//! `tests_cfg_validation.rs` (#1420).

use super::*;
use crate::device::Device;
use crate::dtype::DType;

// -- TimeSchedule tests -------------------------------------------------------

#[test]
fn test_linear_schedule_uniform_spacing() {
    let schedule = TimeSchedule::Linear {
        t_max: 1.0,
        t_min: 0.0,
    };
    let pairs = schedule.steps(4).unwrap();
    assert_eq!(pairs.len(), 4);

    // Linear from 1.0 to 0.0 in 4 steps:
    // t values: 1.0, 0.75, 0.5, 0.25
    // dt values: -0.25 each (stepping downward)
    for (_, dt) in &pairs {
        assert!((dt - (-0.25)).abs() < 1e-6, "dt should be -0.25, got {dt}");
    }
    assert!((pairs[0].0 - 1.0).abs() < 1e-6);
    assert!((pairs[1].0 - 0.75).abs() < 1e-6);
    assert!((pairs[2].0 - 0.5).abs() < 1e-6);
    assert!((pairs[3].0 - 0.25).abs() < 1e-6);
}

#[test]
fn test_linear_schedule_irodori_params() {
    // Irodori uses t_max=0.999, t_min=0.001, 40 steps
    let schedule = TimeSchedule::Linear {
        t_max: 0.999,
        t_min: 0.001,
    };
    let pairs = schedule.steps(40).unwrap();
    assert_eq!(pairs.len(), 40);

    // First timestep should be close to t_max
    assert!((pairs[0].0 - 0.999).abs() < 1e-6);

    // All dt should be equal and negative (stepping down)
    let expected_dt = (0.001 - 0.999) / 40.0;
    for (_, dt) in &pairs {
        assert!(
            (dt - expected_dt).abs() < 1e-5,
            "dt should be ~{expected_dt}, got {dt}"
        );
    }

    // Sum of dt should equal t_min - t_max
    let total_dt: f32 = pairs.iter().map(|(_, dt)| dt).sum();
    assert!(
        (total_dt - (0.001 - 0.999)).abs() < 1e-4,
        "total dt should be -0.998, got {total_dt}"
    );
}

#[test]
fn test_cosine_schedule_values_in_range() {
    let schedule = TimeSchedule::Cosine;
    let pairs = schedule.steps(10).unwrap();
    assert_eq!(pairs.len(), 10);

    // Cosine schedule: t(s) = 1 - cos(s * pi/2), s in [0, 1]
    // t(0) = 1 - cos(0) = 0
    // t(1) = 1 - cos(pi/2) = 1
    // All t values should be in [0, 1]
    for (t, _) in &pairs {
        assert!(*t >= 0.0 && *t <= 1.0, "t={t} out of [0,1]");
    }

    // First timestep is at t(0) = 0.0
    assert!(pairs[0].0.abs() < 1e-6, "first t should be ~0.0");

    // All dt should be positive (stepping upward 0→1)
    for (_, dt) in &pairs {
        assert!(*dt > 0.0, "cosine dt should be positive, got {dt}");
    }
}

#[test]
fn test_cosine_schedule_denser_near_zero() {
    let schedule = TimeSchedule::Cosine;
    let pairs = schedule.steps(20).unwrap();

    // Cosine schedule should have smaller dt near t=0 and larger dt near t=1
    // Compare first dt vs last dt
    let first_dt = pairs[0].1;
    let last_dt = pairs[pairs.len() - 1].1;
    assert!(
        last_dt > first_dt,
        "cosine schedule should have larger steps near t=1: first_dt={first_dt}, last_dt={last_dt}"
    );
}

#[test]
fn test_cosine_schedule_total_coverage() {
    let schedule = TimeSchedule::Cosine;
    let pairs = schedule.steps(10).unwrap();

    // Sum of all dt should equal t(1) - t(0) = 1.0 - 0.0 = 1.0
    let total_dt: f32 = pairs.iter().map(|(_, dt)| dt).sum();
    assert!(
        (total_dt - 1.0).abs() < 1e-5,
        "total dt should be ~1.0, got {total_dt}"
    );
}

#[test]
fn test_schedule_zero_steps_returns_error() {
    let schedule = TimeSchedule::Cosine;
    let result = schedule.steps(0);
    assert!(result.is_err());
}

#[test]
fn test_schedule_single_step() {
    let schedule = TimeSchedule::Cosine;
    let pairs = schedule.steps(1).unwrap();
    assert_eq!(pairs.len(), 1);
    assert!(pairs[0].0.abs() < 1e-6, "single step starts at t=0");
    assert!(
        (pairs[0].1 - 1.0).abs() < 1e-5,
        "single step dt covers full range"
    );
}

// -- Helper: mock velocity field for testing ----------------------------------

/// Constant velocity field — returns the same tensor regardless of state/time.
struct ConstantVelocity {
    velocity: DynTensor,
}

impl VelocityField for ConstantVelocity {
    fn predict(&mut self, _x: &DynTensor, _t: f32) -> Result<DynTensor> {
        Ok(self.velocity.clone())
    }
}

/// Identity velocity field — returns the current state as velocity (v = x).
/// This produces exponential growth: dx/dt = x => x(t) = x0 * e^t.
struct IdentityVelocity;

impl VelocityField for IdentityVelocity {
    fn predict(&mut self, x: &DynTensor, _t: f32) -> Result<DynTensor> {
        Ok(x.clone())
    }
}

/// Velocity field that records the timesteps it was called with.
struct RecordingVelocity {
    timesteps: Vec<f32>,
}

impl VelocityField for RecordingVelocity {
    fn predict(&mut self, x: &DynTensor, t: f32) -> Result<DynTensor> {
        self.timesteps.push(t);
        // Return zeros (no movement)
        DynTensor::zeros(x.dims(), DType::F32, &Device::Cpu)
    }
}

// -- Euler solver tests -------------------------------------------------------

#[test]
fn test_euler_zero_velocity_unchanged() {
    let x0 = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let zeros = DynTensor::zeros(&[3], DType::F32, &Device::Cpu).unwrap();
    let mut vel = ConstantVelocity { velocity: zeros };

    let schedule = TimeSchedule::Linear {
        t_max: 1.0,
        t_min: 0.0,
    };
    let result = euler_solve(&x0, &mut vel, &schedule, 10).unwrap();

    let result_data = result.as_cpu_f32().unwrap();
    assert!((result_data[0] - 1.0).abs() < 1e-6);
    assert!((result_data[1] - 2.0).abs() < 1e-6);
    assert!((result_data[2] - 3.0).abs() < 1e-6);
}

#[test]
fn test_euler_constant_velocity() {
    // With constant velocity v=[1,0,0] and total dt of 1.0 (cosine schedule),
    // x should move by [1,0,0] from initial position.
    let x0 = DynTensor::zeros(&[3], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::new(&[1.0, 0.0, 0.0], &[3], &Device::Cpu).unwrap();
    let mut vel = ConstantVelocity { velocity: v };

    let schedule = TimeSchedule::Cosine;
    let result = euler_solve(&x0, &mut vel, &schedule, 100).unwrap();

    let result_data = result.as_cpu_f32().unwrap();
    // Total displacement = sum(v * dt) = v * sum(dt) = [1,0,0] * 1.0 = [1,0,0]
    assert!(
        (result_data[0] - 1.0).abs() < 1e-4,
        "x[0] should be ~1.0, got {}",
        result_data[0]
    );
    assert!(result_data[1].abs() < 1e-6);
    assert!(result_data[2].abs() < 1e-6);
}

#[test]
fn test_euler_identity_velocity_exponential_growth() {
    // v = x => dx/dt = x => x(t) = x0 * e^t
    // With linear schedule 0→1 (total integration from t=0 to t=1),
    // exact solution at t=1: x = x0 * e
    let x0 = DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap();
    let mut vel = IdentityVelocity;

    // Use linear schedule 0→1 with many steps for accuracy
    // For forward Euler on dx/dt = x from 0 to 1 with N steps:
    // x_N = x_0 * (1 + h)^N where h = 1/N
    // This approaches e as N → ∞
    let schedule = TimeSchedule::Linear {
        t_max: 0.0,
        t_min: 1.0,
    };
    let result = euler_solve(&x0, &mut vel, &schedule, 1000).unwrap();

    let result_data = result.as_cpu_f32().unwrap();
    let expected = std::f32::consts::E;
    // With 1000 steps, Euler should approximate e within ~0.1%
    assert!(
        (result_data[0] - expected).abs() / expected < 0.002,
        "expected ~{expected}, got {}",
        result_data[0]
    );
}

#[test]
fn test_euler_solver_timestep_ordering() {
    let x0 = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let mut vel = RecordingVelocity {
        timesteps: Vec::new(),
    };

    let schedule = TimeSchedule::Cosine;
    let _ = euler_solve(&x0, &mut vel, &schedule, 5).unwrap();

    assert_eq!(vel.timesteps.len(), 5);
    // Cosine schedule timesteps should be monotonically increasing (0→1)
    for i in 1..vel.timesteps.len() {
        assert!(
            vel.timesteps[i] > vel.timesteps[i - 1],
            "timesteps should be increasing: {} <= {}",
            vel.timesteps[i],
            vel.timesteps[i - 1]
        );
    }
}

#[test]
fn test_euler_3d_tensor() {
    // Verify solver works with batch-dimensional 3D tensors
    let x0 = DynTensor::zeros(&[2, 4, 3], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[2, 4, 3], DType::F32, &Device::Cpu).unwrap();
    let mut vel = ConstantVelocity { velocity: v };

    let schedule = TimeSchedule::Linear {
        t_max: 0.0,
        t_min: 1.0,
    };
    // Total dt = 1.0 (from 0 to 1), so x = 0 + 1 * 1 = 1.0 everywhere
    let result = euler_solve(&x0, &mut vel, &schedule, 10).unwrap();

    assert_eq!(result.dims(), &[2, 4, 3]);
    let data = result.as_cpu_f32().unwrap();
    for val in &data {
        assert!((val - 1.0).abs() < 1e-5, "expected ~1.0, got {val}");
    }
}

// CFG solver and NaN/Inf validation tests extracted to
// tests_cfg_validation.rs (#1420).
#[path = "tests_cfg_validation.rs"]
mod cfg_validation_tests;
