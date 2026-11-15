// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Regression tests: a gradient whose shape differs from its parameter is an
//! error, not a panic (or a silent broadcast) inside the optimizer step.
//!
//! `GradStore::var_grads_mut` lets a caller replace any gradient tensor, so
//! the optimizers cannot assume the gradient has the parameter's shape. Before
//! the check, AdamW's fused update indexed the gradient with the parameter's
//! element indices (index out of bounds), and AdaFactor's factored update
//! computed `rank - 1` / `rank - 2` on the gradient's rank (overflow).

use std::sync::Arc;

use crate::adafactor::{AdaFactor, AdaFactorConfig};
use crate::adam::{AdamConfig, AdamW};
use crate::error::OptimError;
use crate::optimizer::Optimizer;
use nn_autodiff::{backward, GradStore, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

/// A 2x2 parameter and the gradient store of `sum(w^2)`, with every gradient
/// then replaced by `replacement`.
fn store_with_grad(replacement: &DynTensor) -> (Var, GradStore) {
    let w = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let loss = t
        .sqr()
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(0)
        .unwrap();
    let mut grads = backward(&loss).unwrap();
    for (_, g) in grads.var_grads_mut() {
        *g = replacement.clone();
    }
    (w, grads)
}

fn assert_shape_mismatch(err: OptimError, got: &[usize]) {
    match err {
        OptimError::GradientShapeMismatch {
            expected,
            got: actual,
        } => {
            assert_eq!(expected, vec![2, 2]);
            assert_eq!(actual, got);
        }
        other => panic!("expected GradientShapeMismatch, got {other:?}"),
    }
}

#[test]
fn adamw_rejects_gradient_with_fewer_elements() {
    let short = DynTensor::from_vec(vec![0.5], &[1], &cpu()).unwrap();
    let (w, grads) = store_with_grad(&short);
    let mut opt = AdamW::new(vec![w.clone()], AdamConfig::default()).unwrap();
    let before = w.data().unwrap().to_vec2::<f32>().unwrap();
    let err = opt.step(&grads).unwrap_err();
    assert_shape_mismatch(err, &[1]);
    // The parameter is untouched.
    assert_eq!(w.data().unwrap().to_vec2::<f32>().unwrap(), before);
}

#[test]
fn adamw_rejects_gradient_with_same_count_other_shape() {
    let flat = DynTensor::from_vec(vec![0.5; 4], &[4], &cpu()).unwrap();
    let (w, grads) = store_with_grad(&flat);
    let mut opt = AdamW::new(vec![w], AdamConfig::default()).unwrap();
    assert_shape_mismatch(opt.step(&grads).unwrap_err(), &[4]);
}

#[test]
fn adafactor_rejects_rank_one_gradient_for_factored_parameter() {
    let flat = DynTensor::from_vec(vec![0.5; 4], &[4], &cpu()).unwrap();
    let (w, grads) = store_with_grad(&flat);
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![w.clone()], config).unwrap();
    let before = w.data().unwrap().to_vec2::<f32>().unwrap();
    assert_shape_mismatch(opt.step(&grads).unwrap_err(), &[4]);
    assert_eq!(w.data().unwrap().to_vec2::<f32>().unwrap(), before);
}

#[test]
fn adafactor_rejects_broadcastable_gradient() {
    // [1, 2] broadcasts against [2, 2]; it must not be applied silently.
    let row = DynTensor::from_vec(vec![0.5, 0.5], &[1, 2], &cpu()).unwrap();
    let (w, grads) = store_with_grad(&row);
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![w], config).unwrap();
    assert_shape_mismatch(opt.step(&grads).unwrap_err(), &[1, 2]);
}

#[test]
fn matching_gradient_still_steps() {
    let good = DynTensor::from_vec(vec![0.5; 4], &[2, 2], &cpu()).unwrap();
    let (w, grads) = store_with_grad(&good);
    let mut opt = AdamW::new(vec![w.clone()], AdamConfig::default()).unwrap();
    opt.step(&grads).unwrap();
    assert_ne!(
        w.data().unwrap().to_vec2::<f32>().unwrap(),
        vec![vec![1.0, 2.0], vec![3.0, 4.0]]
    );
}
