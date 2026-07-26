#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for gradient clipping utilities.

use std::sync::Arc;

use nn_autodiff::{backward, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use super::{clip_grad_norm, clip_grad_value};

/// Helper: create a Var, do a simple forward+backward to populate GradStore.
fn make_grads_with_known_gradient(grad_values: &[f32]) -> nn_autodiff::GradStore {
    // Create var with value 1.0 for each element
    let n = grad_values.len();
    let var = Var::new(DynTensor::from_vec(vec![1.0; n], &[n], &Device::Cpu).unwrap());

    // Use mul_scalar to create a gradient equal to the scalar multiplier.
    // d/dx (x * s) = s, so if we sum, grad = [s, s, ...].
    // Instead, construct gradients directly by doing: loss = sum(x * target_grads)
    // d(loss)/dx_i = target_grads_i
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(grad_values.to_vec(), &[n], &Device::Cpu).unwrap(),
    ));
    let product = t.mul(&target).unwrap();
    let loss = product.sum_keepdim(0).unwrap();
    backward(&loss).unwrap()
}

/// Helper: create multiple vars with known gradients.
fn make_multi_var_grads(grad_sets: &[&[f32]]) -> (Vec<Var>, nn_autodiff::GradStore) {
    let mut vars = Vec::new();
    let mut tracked = Vec::new();
    for grads in grad_sets {
        let n = grads.len();
        let var = Var::new(DynTensor::from_vec(vec![1.0; n], &[n], &Device::Cpu).unwrap());
        let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
        let target = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(grads.to_vec(), &[n], &Device::Cpu).unwrap(),
        ));
        let product = t.mul(&target).unwrap();
        vars.push(var);
        tracked.push(product);
    }

    // Sum all products into a single scalar loss
    let mut total = tracked[0].sum_keepdim(0).unwrap();
    for t in &tracked[1..] {
        let s = t.sum_keepdim(0).unwrap();
        total = total.add(&s).unwrap();
    }
    let grads = backward(&total).unwrap();
    (vars, grads)
}

// ── clip_grad_norm tests ──────────────────────────────────────────

#[test]
fn test_clip_grad_norm_no_clipping_needed() {
    // Gradient = [3.0, 4.0], L2 norm = 5.0, max_norm = 10.0 → no clipping
    let mut grads = make_grads_with_known_gradient(&[3.0, 4.0]);
    let total_norm = clip_grad_norm(&mut grads, 10.0).unwrap();
    assert!((total_norm - 5.0).abs() < 1e-6, "total_norm = {total_norm}");

    // Gradients should be unchanged
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!((vals[0] - 3.0).abs() < 1e-6);
        assert!((vals[1] - 4.0).abs() < 1e-6);
    }
}

#[test]
fn test_clip_grad_norm_clips_when_exceeding() {
    // Gradient = [3.0, 4.0], L2 norm = 5.0, max_norm = 2.5 → scale by 0.5
    let mut grads = make_grads_with_known_gradient(&[3.0, 4.0]);
    let total_norm = clip_grad_norm(&mut grads, 2.5).unwrap();
    assert!((total_norm - 5.0).abs() < 1e-6, "total_norm = {total_norm}");

    // Gradients should be scaled by 2.5/5.0 = 0.5
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!(
            (vals[0] - 1.5).abs() < 1e-5,
            "expected 1.5, got {}",
            vals[0]
        );
        assert!(
            (vals[1] - 2.0).abs() < 1e-5,
            "expected 2.0, got {}",
            vals[1]
        );
    }
}

#[test]
fn test_clip_grad_norm_exact_boundary() {
    // Gradient = [3.0, 4.0], L2 norm = 5.0, max_norm = 5.0 → no clipping
    let mut grads = make_grads_with_known_gradient(&[3.0, 4.0]);
    let total_norm = clip_grad_norm(&mut grads, 5.0).unwrap();
    assert!((total_norm - 5.0).abs() < 1e-6);

    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!((vals[0] - 3.0).abs() < 1e-6);
        assert!((vals[1] - 4.0).abs() < 1e-6);
    }
}

#[test]
fn test_clip_grad_norm_zero_gradients() {
    // All-zero gradients → norm = 0, no clipping
    let mut grads = make_grads_with_known_gradient(&[0.0, 0.0, 0.0]);
    let total_norm = clip_grad_norm(&mut grads, 1.0).unwrap();
    assert!(total_norm.abs() < 1e-10, "total_norm = {total_norm}");
}

#[test]
fn test_clip_grad_norm_multi_var() {
    // Two parameter groups: [3.0] and [4.0] → total L2 = sqrt(9+16) = 5.0
    let (vars, mut grads) = make_multi_var_grads(&[&[3.0], &[4.0]]);
    let total_norm = clip_grad_norm(&mut grads, 2.5).unwrap();
    assert!((total_norm - 5.0).abs() < 1e-5, "total_norm = {total_norm}");

    // Both should be scaled by 0.5
    let g0 = grads.get(&vars[0]).unwrap().to_flat_vec::<f32>().unwrap();
    let g1 = grads.get(&vars[1]).unwrap().to_flat_vec::<f32>().unwrap();
    assert!((g0[0] - 1.5).abs() < 1e-5, "expected 1.5, got {}", g0[0]);
    assert!((g1[0] - 2.0).abs() < 1e-5, "expected 2.0, got {}", g1[0]);
}

#[test]
fn test_clip_grad_norm_invalid_max_norm() {
    let mut grads = make_grads_with_known_gradient(&[1.0]);
    assert!(clip_grad_norm(&mut grads, 0.0).is_err());
    assert!(clip_grad_norm(&mut grads, -1.0).is_err());
    assert!(clip_grad_norm(&mut grads, f64::NAN).is_err());
    assert!(clip_grad_norm(&mut grads, f64::INFINITY).is_err());
}

// ── clip_grad_value tests ─────────────────────────────────────────

#[test]
fn test_clip_grad_value_no_clipping_needed() {
    // All values within range
    let mut grads = make_grads_with_known_gradient(&[0.1, -0.2, 0.3]);
    clip_grad_value(&mut grads, 1.0).unwrap();
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!((vals[0] - 0.1).abs() < 1e-6);
        assert!((vals[1] - (-0.2)).abs() < 1e-6);
        assert!((vals[2] - 0.3).abs() < 1e-6);
    }
}

#[test]
fn test_clip_grad_value_clips_both_directions() {
    // Values exceed range in both directions
    let mut grads = make_grads_with_known_gradient(&[5.0, -5.0, 0.3]);
    clip_grad_value(&mut grads, 1.0).unwrap();
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!(
            (vals[0] - 1.0).abs() < 1e-6,
            "positive clamp: expected 1.0, got {}",
            vals[0]
        );
        assert!(
            (vals[1] - (-1.0)).abs() < 1e-6,
            "negative clamp: expected -1.0, got {}",
            vals[1]
        );
        assert!(
            (vals[2] - 0.3).abs() < 1e-6,
            "within range: expected 0.3, got {}",
            vals[2]
        );
    }
}

#[test]
fn test_clip_grad_value_tight_clip() {
    // Very tight clipping range
    let mut grads = make_grads_with_known_gradient(&[10.0, -10.0]);
    clip_grad_value(&mut grads, 0.01).unwrap();
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!(
            (vals[0] - 0.01).abs() < 1e-6,
            "expected 0.01, got {}",
            vals[0]
        );
        assert!(
            (vals[1] - (-0.01)).abs() < 1e-6,
            "expected -0.01, got {}",
            vals[1]
        );
    }
}

#[test]
fn test_clip_grad_value_multi_var() {
    let (_vars, mut grads) = make_multi_var_grads(&[&[5.0, -5.0], &[0.1, 100.0]]);
    clip_grad_value(&mut grads, 1.0).unwrap();

    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        for &v in &vals {
            assert!(
                (-1.0 - 1e-6..=1.0 + 1e-6).contains(&v),
                "value {v} outside [-1.0, 1.0]"
            );
        }
    }
}

#[test]
fn test_clip_grad_value_invalid_clip_value() {
    let mut grads = make_grads_with_known_gradient(&[1.0]);
    assert!(clip_grad_value(&mut grads, 0.0).is_err());
    assert!(clip_grad_value(&mut grads, -1.0).is_err());
    assert!(clip_grad_value(&mut grads, f64::NAN).is_err());
    assert!(clip_grad_value(&mut grads, f64::INFINITY).is_err());
}

#[test]
fn test_clip_grad_norm_preserves_direction() {
    // After norm clipping, gradient direction (ratios) should be preserved
    let mut grads = make_grads_with_known_gradient(&[6.0, 8.0]);
    clip_grad_norm(&mut grads, 1.0).unwrap();

    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        // Original ratio: 6/8 = 0.75. After scaling: same ratio
        let ratio = vals[0] / vals[1];
        assert!(
            (ratio - 0.75).abs() < 1e-5,
            "direction not preserved: ratio = {ratio}"
        );
        // New norm should be ~1.0
        let norm: f64 = vals
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "clipped norm should be 1.0, got {norm}"
        );
    }
}
