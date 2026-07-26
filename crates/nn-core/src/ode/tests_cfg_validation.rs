// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CFG solver and NaN/Inf validation tests for ODE solvers.
//!
//! Extracted from `tests.rs` for file-size compliance (#1420).

use super::*;
use crate::device::Device;
use crate::dtype::DType;

/// Constant velocity field — returns the same tensor regardless of state/time.
struct ConstantVelocity {
    velocity: DynTensor,
}

impl VelocityField for ConstantVelocity {
    fn predict(&mut self, _x: &DynTensor, _t: f32) -> Result<DynTensor> {
        Ok(self.velocity.clone())
    }
}

// -- CFG solver tests ---------------------------------------------------------

#[test]
fn test_cfg_zero_scale_equals_conditioned() {
    // With cfg_scale=0, result should match conditioned-only solve
    let x0 = DynTensor::zeros(&[3], DType::F32, &Device::Cpu).unwrap();
    let v_cond = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let v_uncond = DynTensor::new(&[10.0, 20.0, 30.0], &[3], &Device::Cpu).unwrap();

    let schedule = TimeSchedule::Cosine;

    let mut cond = ConstantVelocity {
        velocity: v_cond.clone(),
    };
    let mut uncond = ConstantVelocity { velocity: v_uncond };
    let cfg_result = euler_solve_cfg(&x0, &mut cond, &mut uncond, &schedule, 10, 0.0).unwrap();

    let mut cond_only = ConstantVelocity { velocity: v_cond };
    let plain_result = euler_solve(&x0, &mut cond_only, &schedule, 10).unwrap();

    let cfg_data = cfg_result.as_cpu_f32().unwrap();
    let plain_data = plain_result.as_cpu_f32().unwrap();
    for (a, b) in cfg_data.iter().zip(plain_data.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "cfg_scale=0 should match plain solve: {a} vs {b}"
        );
    }
}

#[test]
fn test_cfg_guidance_direction() {
    // CFG should push the result away from unconditioned and toward conditioned.
    // v = v_cond + cfg * (v_cond - v_uncond)
    // With v_cond=2, v_uncond=1, cfg=1: v = 2 + 1*(2-1) = 3
    // With v_cond=2, v_uncond=1, cfg=0: v = 2
    let x0 = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let v_cond = DynTensor::new(&[2.0], &[1], &Device::Cpu).unwrap();
    let v_uncond = DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap();

    let schedule = TimeSchedule::Cosine;

    // cfg_scale=0: effective v = 2.0
    let mut c0 = ConstantVelocity {
        velocity: v_cond.clone(),
    };
    let mut u0 = ConstantVelocity {
        velocity: v_uncond.clone(),
    };
    let no_cfg = euler_solve_cfg(&x0, &mut c0, &mut u0, &schedule, 50, 0.0).unwrap();

    // cfg_scale=1: effective v = 3.0
    let mut c1 = ConstantVelocity { velocity: v_cond };
    let mut u1 = ConstantVelocity { velocity: v_uncond };
    let with_cfg = euler_solve_cfg(&x0, &mut c1, &mut u1, &schedule, 50, 1.0).unwrap();

    let no_cfg_val = no_cfg.as_cpu_f32().unwrap()[0];
    let with_cfg_val = with_cfg.as_cpu_f32().unwrap()[0];

    // Total dt = 1.0 for cosine schedule
    // no_cfg: x = 0 + 2.0 * 1.0 = 2.0
    // with_cfg: x = 0 + 3.0 * 1.0 = 3.0
    assert!(
        (no_cfg_val - 2.0).abs() < 0.01,
        "no_cfg expected ~2.0, got {no_cfg_val}"
    );
    assert!(
        (with_cfg_val - 3.0).abs() < 0.01,
        "with_cfg expected ~3.0, got {with_cfg_val}"
    );
    assert!(
        with_cfg_val > no_cfg_val,
        "CFG should amplify: {with_cfg_val} > {no_cfg_val}"
    );
}

#[test]
fn test_cfg_cosyvoice_scale() {
    // CosyVoice3 uses cfg_scale=0.7
    // v = v_cond + 0.7 * (v_cond - v_uncond)
    // With v_cond=1.0, v_uncond=0.0: v = 1.0 + 0.7 * 1.0 = 1.7
    let x0 = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let v_cond = DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap();
    let v_uncond = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();

    let schedule = TimeSchedule::Cosine;

    let mut c = ConstantVelocity { velocity: v_cond };
    let mut u = ConstantVelocity { velocity: v_uncond };
    let result = euler_solve_cfg(&x0, &mut c, &mut u, &schedule, 100, 0.7).unwrap();

    let val = result.as_cpu_f32().unwrap()[0];
    // Total dt = 1.0, so x = 0 + 1.7 * 1.0 = 1.7
    assert!(
        (val - 1.7).abs() < 0.01,
        "CosyVoice cfg=0.7 expected ~1.7, got {val}"
    );
}

#[test]
fn test_euler_single_step() {
    // Single Euler step: x1 = x0 + v * dt
    let x0 = DynTensor::new(&[0.0], &[1], &Device::Cpu).unwrap();
    let v = DynTensor::new(&[5.0], &[1], &Device::Cpu).unwrap();
    let mut vel = ConstantVelocity { velocity: v };

    let schedule = TimeSchedule::Linear {
        t_max: 0.0,
        t_min: 1.0,
    };
    let result = euler_solve(&x0, &mut vel, &schedule, 1).unwrap();

    let val = result.as_cpu_f32().unwrap()[0];
    // dt = (1.0 - 0.0) / 1 = 1.0, so x = 0 + 5 * 1.0 = 5.0
    assert!(
        (val - 5.0).abs() < 1e-5,
        "single step expected 5.0, got {val}"
    );
}

// -- NaN/Inf input validation tests -------------------------------------------

#[test]
fn test_linear_schedule_nan_t_max_returns_error() {
    let schedule = TimeSchedule::Linear {
        t_max: f32::NAN,
        t_min: 0.0,
    };
    let result = schedule.steps(10);
    assert!(result.is_err(), "NaN t_max should be rejected");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("finite"),
        "error should mention finiteness: {msg}"
    );
}

#[test]
fn test_linear_schedule_inf_t_min_returns_error() {
    let schedule = TimeSchedule::Linear {
        t_max: 1.0,
        t_min: f32::INFINITY,
    };
    let result = schedule.steps(10);
    assert!(result.is_err(), "Inf t_min should be rejected");
}

#[test]
fn test_linear_schedule_neg_inf_returns_error() {
    let schedule = TimeSchedule::Linear {
        t_max: f32::NEG_INFINITY,
        t_min: 0.0,
    };
    let result = schedule.steps(10);
    assert!(result.is_err(), "NEG_INFINITY t_max should be rejected");
}

#[test]
fn test_cfg_nan_scale_returns_error() {
    let x0 = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap();
    let mut cond = ConstantVelocity {
        velocity: v.clone(),
    };
    let mut uncond = ConstantVelocity { velocity: v };
    let schedule = TimeSchedule::Cosine;

    let result = euler_solve_cfg(&x0, &mut cond, &mut uncond, &schedule, 10, f32::NAN);
    assert!(result.is_err(), "NaN cfg_scale should be rejected");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("finite"),
        "error should mention finiteness: {msg}"
    );
}

#[test]
fn test_cfg_inf_scale_returns_error() {
    let x0 = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap();
    let mut cond = ConstantVelocity {
        velocity: v.clone(),
    };
    let mut uncond = ConstantVelocity { velocity: v };
    let schedule = TimeSchedule::Cosine;

    let result = euler_solve_cfg(&x0, &mut cond, &mut uncond, &schedule, 10, f32::INFINITY);
    assert!(result.is_err(), "Inf cfg_scale should be rejected");
}
