#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for DynTensor activation-style math operations: leaky_relu, snake, snake_tensor,
//! and repair_non_finite device regression test. Extracted from `tests_math.rs` for 500-line compliance.

use crate::dyn_tensor::test_helpers::{cpu, t1d};
use crate::DynTensor;

// -- leaky_relu ---------------------------------------------------------------

#[test]
fn test_leaky_relu_positive_unchanged() {
    let x = t1d(&[1.0, 2.0, 3.0]);
    let y = x.leaky_relu(0.01).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_leaky_relu_negative_scaled() {
    let x = t1d(&[-10.0, -1.0, 0.0, 1.0, 10.0]);
    let y = x.leaky_relu(0.1).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - (-1.0)).abs() < 1e-5, "got {}", vals[0]);
    assert!((vals[1] - (-0.1)).abs() < 1e-5, "got {}", vals[1]);
    assert!((vals[2] - 0.0).abs() < 1e-5, "got {}", vals[2]);
    assert!((vals[3] - 1.0).abs() < 1e-5, "got {}", vals[3]);
    assert!((vals[4] - 10.0).abs() < 1e-5, "got {}", vals[4]);
}

#[test]
fn test_leaky_relu_2d_shape_preserved() {
    let x = DynTensor::from_vec(vec![-2.0, 1.0, -0.5, 3.0], &[2, 2], &cpu()).unwrap();
    let y = x.leaky_relu(0.2).unwrap();
    assert_eq!(y.dims(), &[2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - (-0.4)).abs() < 1e-5); // -2.0 * 0.2
    assert!((vals[1] - 1.0).abs() < 1e-5);
    assert!((vals[2] - (-0.1)).abs() < 1e-5); // -0.5 * 0.2
    assert!((vals[3] - 3.0).abs() < 1e-5);
}

// -- snake --------------------------------------------------------------------

#[test]
fn test_snake_zero_input() {
    // snake(0, alpha) = 0 + (1/alpha) * sin²(0) = 0
    let x = t1d(&[0.0]);
    let y = x.snake(1.0).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(
        (vals[0] - 0.0).abs() < 1e-6,
        "snake(0) should be 0, got {}",
        vals[0]
    );
}

#[test]
fn test_snake_known_value() {
    // snake(x, alpha) = x + (1/alpha) * sin²(alpha * x)
    // For x=1.0, alpha=1.0: snake = 1.0 + sin²(1.0) = 1.0 + 0.7080734... ≈ 1.7081
    let x = t1d(&[1.0]);
    let y = x.snake(1.0).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    let expected = 1.0 + (1.0_f32).sin().powi(2);
    assert!(
        (vals[0] - expected).abs() < 1e-4,
        "snake(1, alpha=1) should be ~{expected}, got {}",
        vals[0]
    );
}

#[test]
fn test_snake_always_ge_input() {
    // snake(x, alpha) = x + non_negative_term ≥ x for all x
    let x = t1d(&[-3.0, -1.0, 0.0, 1.0, 3.0]);
    let y = x.snake(2.0).unwrap();
    let x_vals = x.to_flat_vec::<f32>().unwrap();
    let y_vals = y.to_flat_vec::<f32>().unwrap();
    for (i, (&xv, &yv)) in x_vals.iter().zip(y_vals.iter()).enumerate() {
        assert!(
            yv >= xv - 1e-6,
            "snake(x) should be >= x: x[{i}]={xv}, snake={yv}"
        );
    }
}

#[test]
fn test_snake_alpha_clamped() {
    // Tiny alpha should be clamped to 1e-6, not divide by zero
    let x = t1d(&[1.0]);
    let y = x.snake(0.0).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(
        vals[0].is_finite(),
        "snake with alpha=0 should produce finite output"
    );
}

// -- snake_tensor -------------------------------------------------------------

#[test]
fn test_snake_tensor_matches_scalar() {
    // Per-channel snake with uniform alpha should match scalar snake
    let x = t1d(&[1.0, 2.0, 3.0]);
    let alpha = DynTensor::from_vec(vec![2.0, 2.0, 2.0], &[3], &cpu()).unwrap();
    let y_tensor = x.snake_tensor(&alpha).unwrap();
    let y_scalar = x.snake(2.0).unwrap();
    let tv = y_tensor.to_flat_vec::<f32>().unwrap();
    let sv = y_scalar.to_flat_vec::<f32>().unwrap();
    for i in 0..3 {
        assert!(
            (tv[i] - sv[i]).abs() < 1e-4,
            "mismatch at [{i}]: tensor={}, scalar={}",
            tv[i],
            sv[i]
        );
    }
}

#[test]
fn test_snake_tensor_per_channel() {
    // [1, C, T] input with [1, C, 1] alpha — different alpha per channel
    let x = DynTensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 2, 2], &cpu()).unwrap();
    let alpha = DynTensor::from_vec(vec![1.0, 4.0], &[1, 2, 1], &cpu()).unwrap();
    let y = x.snake_tensor(&alpha).unwrap();
    assert_eq!(y.dims(), &[1, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Channel 0 (alpha=1): snake(1,1) = 1 + sin²(1) ≈ 1.708
    let expected_ch0 = 1.0 + (1.0_f32).sin().powi(2);
    // Channel 1 (alpha=4): snake(1,4) = 1 + (1/4) * sin²(4) ≈ 1 + 0.25*sin²(4)
    let expected_ch1 = 1.0 + 0.25 * (4.0_f32).sin().powi(2);
    assert!(
        (vals[0] - expected_ch0).abs() < 1e-4,
        "ch0: expected ~{expected_ch0}, got {}",
        vals[0]
    );
    assert!(
        (vals[2] - expected_ch1).abs() < 1e-4,
        "ch1: expected ~{expected_ch1}, got {}",
        vals[2]
    );
}

#[test]
fn test_snake_tensor_alpha_clamped() {
    // Alpha with a zero should be clamped to 1e-6
    let x = t1d(&[1.0, 2.0]);
    let alpha = DynTensor::from_vec(vec![0.0, 3.0], &[2], &cpu()).unwrap();
    let y = x.snake_tensor(&alpha).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(
        vals[0].is_finite(),
        "zero alpha should produce finite output"
    );
    assert!(vals[1].is_finite());
}

// -- softplus ------------------------------------------------------------------

#[test]
fn test_softplus_basic_values() {
    // softplus(x) = log(1 + exp(x))
    let x = t1d(&[0.0, 1.0, -1.0]);
    let y = x.softplus().unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    let expected = [
        0.0_f32.exp().ln_1p(), // log(2) ≈ 0.6931
        1.0_f32.exp().ln_1p(),     // log(1+e) ≈ 1.3133
        (-1.0_f32).exp().ln_1p(),  // log(1+1/e) ≈ 0.3133
    ];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "softplus[{i}]: expected {exp}, got {got}"
        );
    }
}

#[test]
fn test_softplus_always_positive() {
    // softplus(x) >= 0 for all x (theoretically > 0, but f32 rounds
    // log(1+exp(-100)) to 0.0 due to exp(-100) being subnormal)
    let x = t1d(&[-10.0, -1.0, 0.0, 1.0, 10.0]);
    let y = x.softplus().unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(v >= 0.0, "softplus[{i}] should be >= 0, got {v}");
    }
    // For moderate inputs, softplus should be strictly positive
    assert!(
        vals[0] > 0.0,
        "softplus(-10) should be > 0, got {}",
        vals[0]
    );
    assert!(vals[1] > 0.0, "softplus(-1) should be > 0, got {}", vals[1]);
}

#[test]
fn test_softplus_large_input_approx_identity() {
    // For large x, softplus(x) ≈ x
    let x = t1d(&[20.0, 50.0]);
    let y = x.softplus().unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(
        (vals[0] - 20.0).abs() < 1e-4,
        "softplus(20) ≈ 20, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 50.0).abs() < 1e-4,
        "softplus(50) ≈ 50, got {}",
        vals[1]
    );
}

// -- selu ---------------------------------------------------------------------

#[test]
fn test_selu_positive_scaled() {
    // For x >= 0, selu(x) = lambda * x where lambda ≈ 1.0507
    let x = t1d(&[0.0, 1.0, 2.0]);
    let y = x.selu().unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    let lambda: f32 = 1.050_701;
    assert!((vals[0] - 0.0).abs() < 1e-6, "selu(0) = 0, got {}", vals[0]);
    assert!(
        (vals[1] - lambda).abs() < 1e-4,
        "selu(1) ≈ {lambda}, got {}",
        vals[1]
    );
    assert!(
        (vals[2] - 2.0 * lambda).abs() < 1e-4,
        "selu(2) ≈ {}, got {}",
        2.0 * lambda,
        vals[2]
    );
}

#[test]
fn test_selu_negative_values() {
    // For x < 0, selu(x) = lambda * alpha * (exp(x) - 1)
    let x = t1d(&[-1.0, -5.0]);
    let y = x.selu().unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    let alpha: f32 = 1.673_263_2;
    let lambda: f32 = 1.050_701;
    let expected_m1 = lambda * alpha * (-1.0_f32).exp_m1();
    assert!(
        (vals[0] - expected_m1).abs() < 1e-4,
        "selu(-1) ≈ {expected_m1}, got {}",
        vals[0]
    );
    // selu saturates around -lambda*alpha ≈ -1.7581 for large negative x
    assert!(
        vals[1] > -2.0 && vals[1] < 0.0,
        "selu(-5) should be in (-2, 0), got {}",
        vals[1]
    );
}

// -- celu ---------------------------------------------------------------------

#[test]
fn test_celu_positive_unchanged() {
    // For x >= 0, celu(x) = x
    let x = t1d(&[0.0, 1.0, 5.0]);
    let y = x.celu(1.0).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 0.0).abs() < 1e-6);
    assert!((vals[1] - 1.0).abs() < 1e-6);
    assert!((vals[2] - 5.0).abs() < 1e-6);
}

#[test]
fn test_celu_negative_values() {
    // For x < 0, celu(x, alpha) = alpha * (exp(x/alpha) - 1)
    let x = t1d(&[-1.0, -2.0]);
    let y = x.celu(1.0).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    let expected_m1 = 1.0_f32 * (-1.0_f32 / 1.0).exp_m1();
    assert!(
        (vals[0] - expected_m1).abs() < 1e-5,
        "celu(-1, alpha=1) ≈ {expected_m1}, got {}",
        vals[0]
    );
}

#[test]
fn test_celu_alpha_parameter() {
    // Different alpha values change the negative-side slope
    let x = t1d(&[-1.0]);
    let y1 = x.celu(0.5).unwrap();
    let y2 = x.celu(2.0).unwrap();
    let v1 = y1.to_flat_vec::<f32>().unwrap()[0];
    let v2 = y2.to_flat_vec::<f32>().unwrap()[0];
    // alpha=0.5: 0.5*(exp(-1/0.5)-1) = 0.5*(exp(-2)-1) ≈ -0.4323
    // alpha=2.0: 2.0*(exp(-1/2)-1) = 2.0*(exp(-0.5)-1) ≈ -0.7869
    let exp_05 = 0.5_f32 * (-1.0_f32 / 0.5).exp_m1();
    let exp_20 = 2.0_f32 * (-1.0_f32 / 2.0).exp_m1();
    assert!(
        (v1 - exp_05).abs() < 1e-4,
        "celu(-1, 0.5) ≈ {exp_05}, got {v1}"
    );
    assert!(
        (v2 - exp_20).abs() < 1e-4,
        "celu(-1, 2.0) ≈ {exp_20}, got {v2}"
    );
}

// -- hard_sigmoid -------------------------------------------------------------

#[test]
fn test_hard_sigmoid_known_values() {
    // hard_sigmoid(x) = max(0, min(1, x/6 + 0.5))
    let x = t1d(&[-4.0, -3.0, 0.0, 3.0, 4.0]);
    let y = x.hard_sigmoid().unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(
        (vals[0] - 0.0).abs() < 1e-6,
        "hard_sigmoid(-4) = 0, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 0.0).abs() < 1e-6,
        "hard_sigmoid(-3) = 0, got {}",
        vals[1]
    );
    assert!(
        (vals[2] - 0.5).abs() < 1e-6,
        "hard_sigmoid(0) = 0.5, got {}",
        vals[2]
    );
    assert!(
        (vals[3] - 1.0).abs() < 1e-6,
        "hard_sigmoid(3) = 1, got {}",
        vals[3]
    );
    assert!(
        (vals[4] - 1.0).abs() < 1e-6,
        "hard_sigmoid(4) = 1, got {}",
        vals[4]
    );
}

#[test]
fn test_hard_sigmoid_output_range() {
    // Output is always in [0, 1]
    let x = t1d(&[-100.0, -1.0, 0.0, 1.0, 100.0]);
    let y = x.hard_sigmoid().unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&v),
            "hard_sigmoid[{i}] should be in [0,1], got {v}"
        );
    }
}

// -- hard_swish ---------------------------------------------------------------

#[test]
fn test_hard_swish_known_values() {
    // hard_swish(x) = x * hard_sigmoid(x) = x * max(0, min(1, x/6 + 0.5))
    let x = t1d(&[-4.0, 0.0, 3.0, 6.0]);
    let y = x.hard_swish().unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // x=-4: -4 * 0 = 0
    assert!(
        (vals[0] - 0.0).abs() < 1e-6,
        "hard_swish(-4) = 0, got {}",
        vals[0]
    );
    // x=0: 0 * 0.5 = 0
    assert!(
        (vals[1] - 0.0).abs() < 1e-6,
        "hard_swish(0) = 0, got {}",
        vals[1]
    );
    // x=3: 3 * 1.0 = 3
    assert!(
        (vals[2] - 3.0).abs() < 1e-6,
        "hard_swish(3) = 3, got {}",
        vals[2]
    );
    // x=6: 6 * 1.0 = 6
    assert!(
        (vals[3] - 6.0).abs() < 1e-6,
        "hard_swish(6) = 6, got {}",
        vals[3]
    );
}

#[test]
fn test_hard_swish_negative_region() {
    // In the linear region (-3, 3), hard_swish(x) = x*(x/6 + 0.5)
    let x = t1d(&[-2.0, -1.0, 1.0, 2.0]);
    let y = x.hard_swish().unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for (i, &xv) in [-2.0_f32, -1.0, 1.0, 2.0].iter().enumerate() {
        let expected = xv * (xv / 6.0 + 0.5).clamp(0.0, 1.0);
        assert!(
            (vals[i] - expected).abs() < 1e-5,
            "hard_swish({xv}) ≈ {expected}, got {}",
            vals[i]
        );
    }
}

// -- mish ---------------------------------------------------------------------

#[test]
fn test_mish_known_values() {
    // mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    let x = t1d(&[0.0, 1.0, -1.0]);
    let y = x.mish().unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for (i, &xv) in [0.0_f32, 1.0, -1.0].iter().enumerate() {
        let sp = xv.exp().ln_1p();
        let expected = xv * sp.tanh();
        assert!(
            (vals[i] - expected).abs() < 1e-5,
            "mish({xv}) ≈ {expected}, got {}",
            vals[i]
        );
    }
}

#[test]
fn test_mish_zero_returns_zero() {
    let x = t1d(&[0.0]);
    let y = x.mish().unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 0.0).abs() < 1e-6, "mish(0) = 0, got {}", vals[0]);
}

#[test]
fn test_mish_shape_preserved() {
    let x = DynTensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0, -2.0, 3.0], &[2, 3], &cpu()).unwrap();
    let y = x.mish().unwrap();
    assert_eq!(y.dims(), &[2, 3]);
}

// -- elu edge cases -----------------------------------------------------------

#[test]
fn test_elu_known_values_cpu() {
    // elu(x) = x if x > 0, else alpha * (exp(x) - 1)
    let x = t1d(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let y = x.elu(1.0).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    let expected = [
        1.0_f32 * (-2.0_f32).exp_m1(),
        1.0_f32 * (-1.0_f32).exp_m1(),
        0.0,
        1.0,
        2.0,
    ];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "elu[{i}]: expected {exp}, got {got}"
        );
    }
}

#[test]
fn test_elu_alpha_parameter() {
    // Different alpha changes the negative-side magnitude
    let x = t1d(&[-1.0]);
    let y1 = x.elu(0.5).unwrap();
    let y2 = x.elu(2.0).unwrap();
    let v1 = y1.to_flat_vec::<f32>().unwrap()[0];
    let v2 = y2.to_flat_vec::<f32>().unwrap()[0];
    let exp_05 = 0.5_f32 * (-1.0_f32).exp_m1();
    let exp_20 = 2.0_f32 * (-1.0_f32).exp_m1();
    assert!(
        (v1 - exp_05).abs() < 1e-5,
        "elu(-1, 0.5) ≈ {exp_05}, got {v1}"
    );
    assert!(
        (v2 - exp_20).abs() < 1e-5,
        "elu(-1, 2.0) ≈ {exp_20}, got {v2}"
    );
}

// -- AC1 regression: repair_non_finite must work on CPU (GPU round-trip logic) --

#[test]
fn test_repair_non_finite_preserves_device_cpu() {
    // Verify repair_non_finite returns a CPU tensor when input is CPU.
    let x = DynTensor::from_vec(vec![1.0, f32::NAN, 3.0, f32::INFINITY], &[4], &cpu()).unwrap();
    let repaired = x.repair_non_finite(0.0).unwrap();
    assert_eq!(repaired.device(), cpu());
    assert_eq!(
        repaired.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 0.0, 3.0, 0.0]
    );
}
