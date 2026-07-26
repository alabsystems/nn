// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: autodiff backward pass feeding into nn-optim optimizers.
//!
//! Covers:
//! - Forward -> backward -> optimizer step on a simple linear model
//! - Loss decrease over multiple SGD steps on a quadratic
//! - Loss decrease over multiple Adam steps on a quadratic
//! - Gradient accumulation across multiple forward passes
//! - Fresh forward required after optimizer step
//! - Mixed dtype gradient handling (BF16 forward, F32 grads)
//! - Gradient clipping integration (clip_grad_norm)
//! - AdamW weight decay effect
//! - Optimizer backward_step convenience
//! - Learning rate change mid-training
//! - Multi-variable model convergence
//! - SGD momentum convergence
//! - Matmul-based linear layer optimization
//! - clip_grad_value integration

use std::sync::Arc;

use nn_autodiff::{backward, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_optim::grad_clip::{clip_grad_norm, clip_grad_value};
use nn_optim::optimizer::Optimizer;
use nn_optim::{AdamConfig, AdamW, Sgd, SgdConfig};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cpu() -> Device {
    Device::Cpu
}

/// Create an SgdConfig, working around #[non_exhaustive] cross-crate restriction.
fn sgd_config(lr: f64, momentum: f64, weight_decay: f64) -> SgdConfig {
    let mut config = SgdConfig::default();
    config.lr = lr;
    config.momentum = momentum;
    config.weight_decay = weight_decay;
    config
}

/// Create an AdamConfig, working around #[non_exhaustive] cross-crate restriction.
fn adam_config(lr: f64, weight_decay: f64) -> AdamConfig {
    let mut config = AdamConfig::default();
    config.lr = lr;
    config.weight_decay = weight_decay;
    config
}

fn var_from(data: Vec<f32>, shape: &[usize]) -> Var {
    Var::new(DynTensor::from_vec(data, shape, &cpu()).unwrap())
}

fn tracked(v: &Var) -> Arc<TrackedTensor> {
    Arc::new(TrackedTensor::from_var(v).unwrap())
}

/// Build a scalar loss = sum(elements) from an arbitrary-shape tracked tensor.
fn scalar_loss(t: &Arc<TrackedTensor>) -> Arc<TrackedTensor> {
    let mut result = Arc::clone(t);
    for d in (0..result.tensor().rank()).rev() {
        result = result.sum_keepdim(d).unwrap();
    }
    result
}

fn get_scalar(t: &DynTensor) -> f32 {
    t.to_scalar::<f32>().unwrap()
}

/// Compute quadratic loss: loss = sum((w * x - target)^2)
/// where x and target are constants, w is trainable.
fn quadratic_loss(w: &Var, x: &DynTensor, target: &DynTensor) -> Arc<TrackedTensor> {
    let tw = tracked(w);
    let tx = Arc::new(TrackedTensor::from_tensor(x.clone()));
    let tt = Arc::new(TrackedTensor::from_tensor(target.clone()));
    let pred = tw.mul(&tx).unwrap();
    let diff = pred.sub(&tt).unwrap();
    let sq = diff.sqr().unwrap();
    scalar_loss(&sq)
}

/// Evaluate quadratic loss value (forward only, no tracking).
fn eval_quadratic_loss(w: &Var, x: &DynTensor, target: &DynTensor) -> f32 {
    let w_data = w.data().unwrap();
    let pred = w_data.mul(x).unwrap();
    let diff = pred.sub(target).unwrap();
    let sq = diff.sqr().unwrap();
    let mut result = sq;
    for d in (0..result.rank()).rev() {
        result = result.sum_keepdim(d).unwrap();
    }
    get_scalar(&result)
}

// ===========================================================================
// 1. Forward -> backward -> optimizer step on simple linear model
// ===========================================================================

#[test]
fn test_forward_backward_sgd_step_linear() {
    // w starts at 1.0, target is 3.0, x is 1.0
    // loss = (w*x - target)^2 = (1 - 3)^2 = 4
    // d(loss)/dw = 2*(w*x - target)*x = 2*(1-3)*1 = -4
    // w_new = w - lr * grad = 1 - 0.1*(-4) = 1.4
    let w = var_from(vec![1.0], &[1]);
    let x = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    let target = DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap();

    let mut sgd = Sgd::new(vec![w.clone()], sgd_config(0.1, 0.0, 0.0)).unwrap();

    let loss = quadratic_loss(&w, &x, &target);
    let grads = backward(&loss).unwrap();
    sgd.step(&grads).unwrap();

    let w_after = w.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (w_after[0] - 1.4).abs() < 1e-5,
        "expected w=1.4 after SGD step, got {}",
        w_after[0]
    );
}

// ===========================================================================
// 2. Loss decreases over multiple SGD steps on quadratic
// ===========================================================================

#[test]
fn test_sgd_loss_decreases_quadratic() {
    let w = var_from(vec![0.0, 0.0, 0.0], &[3]);
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let target = DynTensor::from_vec(vec![2.0, 4.0, 6.0], &[3], &cpu()).unwrap();

    let mut sgd = Sgd::new(vec![w.clone()], sgd_config(0.01, 0.0, 0.0)).unwrap();

    let initial_loss = eval_quadratic_loss(&w, &x, &target);

    for _ in 0..50 {
        let loss_tracked = quadratic_loss(&w, &x, &target);
        let grads = backward(&loss_tracked).unwrap();
        sgd.step(&grads).unwrap();
    }

    let final_loss = eval_quadratic_loss(&w, &x, &target);
    assert!(
        final_loss < initial_loss * 0.5,
        "SGD should reduce loss by >50% over 50 steps: initial={initial_loss}, final={final_loss}"
    );
}

// ===========================================================================
// 3. Loss decreases over multiple Adam steps on quadratic
// ===========================================================================

#[test]
fn test_adam_loss_decreases_quadratic() {
    let w = var_from(vec![0.0, 0.0], &[2]);
    let x = DynTensor::from_vec(vec![1.0, 1.0], &[2], &cpu()).unwrap();
    let target = DynTensor::from_vec(vec![3.0, 5.0], &[2], &cpu()).unwrap();

    let mut adam = AdamW::new(vec![w.clone()], adam_config(0.1, 0.0)).unwrap();

    let initial_loss = eval_quadratic_loss(&w, &x, &target);

    for _ in 0..80 {
        let loss_tracked = quadratic_loss(&w, &x, &target);
        let grads = backward(&loss_tracked).unwrap();
        adam.step(&grads).unwrap();
    }

    let final_loss = eval_quadratic_loss(&w, &x, &target);
    assert!(
        final_loss < initial_loss * 0.01,
        "Adam should reduce loss by >99%: initial={initial_loss}, final={final_loss}"
    );
}

// ===========================================================================
// 4. Gradient accumulation across multiple forward passes
// ===========================================================================

#[test]
fn test_gradient_accumulation_two_passes() {
    // Accumulate gradients from two different inputs before stepping.
    // w = [1.0], x1 = [2.0], x2 = [3.0]
    // loss1 = (w*x1)^2 = 4, d/dw = 2*w*x1^2 = 8
    // loss2 = (w*x2)^2 = 9, d/dw = 2*w*x2^2 = 18
    // Combined loss = 4 + 9 = 13, d/dw = 8 + 18 = 26
    let w = var_from(vec![1.0], &[1]);
    let x1 = DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap();
    let x2 = DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap();

    let tw1 = tracked(&w);
    let tx1 = Arc::new(TrackedTensor::from_tensor(x1));
    let pred1 = tw1.mul(&tx1).unwrap();
    let sq1 = pred1.sqr().unwrap();

    let tw2 = tracked(&w);
    let tx2 = Arc::new(TrackedTensor::from_tensor(x2));
    let pred2 = tw2.mul(&tx2).unwrap();
    let sq2 = pred2.sqr().unwrap();

    let total = sq1.add(&sq2).unwrap();
    let loss = scalar_loss(&total);
    let grads = backward(&loss).unwrap();

    let grad_w = grads.get(&w).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad_w[0] - 26.0).abs() < 1e-4,
        "expected accumulated grad=26.0, got {}",
        grad_w[0]
    );
}

// ===========================================================================
// 5. Fresh forward required after step
// ===========================================================================

#[test]
fn test_stale_grad_not_reused() {
    let w = var_from(vec![1.0], &[1]);
    let x = DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap();
    let target = DynTensor::from_vec(vec![4.0], &[1], &cpu()).unwrap();

    let mut sgd = Sgd::new(vec![w.clone()], sgd_config(0.5, 0.0, 0.0)).unwrap();

    // First forward+backward+step
    let loss1 = quadratic_loss(&w, &x, &target);
    let grads1 = backward(&loss1).unwrap();
    let grad1_val = grads1.get(&w).unwrap().to_flat_vec::<f32>().unwrap()[0];
    sgd.step(&grads1).unwrap();

    // w changed: re-do forward to get correct gradient
    let loss2 = quadratic_loss(&w, &x, &target);
    let grads2 = backward(&loss2).unwrap();
    let grad2_val = grads2.get(&w).unwrap().to_flat_vec::<f32>().unwrap()[0];

    assert!(
        (grad1_val - grad2_val).abs() > 1e-3,
        "gradients should differ after step: grad1={grad1_val}, grad2={grad2_val}"
    );
}

// ===========================================================================
// 6. Mixed dtype: BF16 var, F32 gradients
// ===========================================================================

#[test]
fn test_bf16_var_f32_gradients() {
    let bf16_data = DynTensor::zeros(&[3], DType::BF16, &cpu()).unwrap();
    let w = Var::new(bf16_data);

    let tw = tracked(&w);
    let sq = tw.sqr().unwrap();
    let loss = scalar_loss(&sq);
    let grads = backward(&loss).unwrap();

    let grad = grads.get(&w).unwrap();
    assert_eq!(
        grad.dtype(),
        w.dtype().unwrap(),
        "gradient dtype should match variable dtype"
    );
    assert_eq!(grad.dims(), &[3]);
}

// ===========================================================================
// 7. Gradient clipping integration (clip_grad_norm)
// ===========================================================================

#[test]
fn test_clip_grad_norm_then_step() {
    let w = var_from(vec![100.0, 200.0, 300.0], &[3]);
    let x = DynTensor::from_vec(vec![1.0, 1.0, 1.0], &[3], &cpu()).unwrap();
    let target = DynTensor::from_vec(vec![0.0, 0.0, 0.0], &[3], &cpu()).unwrap();

    let mut sgd = Sgd::new(vec![w.clone()], sgd_config(0.01, 0.0, 0.0)).unwrap();

    let loss = quadratic_loss(&w, &x, &target);
    let mut grads = backward(&loss).unwrap();

    let original_norm = clip_grad_norm(&mut grads, 1.0).unwrap();
    assert!(
        original_norm > 1.0,
        "expected large gradient norm, got {original_norm}"
    );

    // After clipping, gradients should have norm ~= 1.0
    let mut clipped_norm_sq = 0.0f64;
    for (_id, g) in grads.var_grads() {
        let norm_sq = g
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        clipped_norm_sq += f64::from(norm_sq);
    }
    let clipped_norm = clipped_norm_sq.sqrt();
    assert!(
        (clipped_norm - 1.0).abs() < 1e-4,
        "expected clipped norm ~1.0, got {clipped_norm}"
    );

    sgd.step(&grads).unwrap();
}

// ===========================================================================
// 8. AdamW weight decay effect
// ===========================================================================

#[test]
fn test_adam_weight_decay_shrinks_weights() {
    let w = var_from(vec![10.0, 20.0], &[2]);

    let mut adam = AdamW::new(vec![w.clone()], adam_config(0.01, 0.1)).unwrap();

    let initial_norm: f32 = w
        .data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();

    // Step with near-zero gradient (only weight decay applies)
    let tw = tracked(&w);
    let loss = tw.mul_scalar(1e-10).unwrap();
    let loss = scalar_loss(&loss);
    let grads = backward(&loss).unwrap();
    adam.step(&grads).unwrap();

    let final_norm: f32 = w
        .data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();

    assert!(
        final_norm < initial_norm,
        "weight decay should shrink weights: initial_norm={initial_norm}, final_norm={final_norm}"
    );
}

// ===========================================================================
// 9. Optimizer backward_step convenience
// ===========================================================================

#[test]
fn test_backward_step_convenience() {
    let w = var_from(vec![0.0], &[1]);
    let x = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    let target = DynTensor::from_vec(vec![5.0], &[1], &cpu()).unwrap();

    let mut sgd = Sgd::new(vec![w.clone()], sgd_config(0.1, 0.0, 0.0)).unwrap();

    let initial = w.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    let loss = quadratic_loss(&w, &x, &target);
    sgd.backward_step(&loss).unwrap();

    let after = w.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (initial - after).abs() > 1e-6,
        "backward_step should update w: before={initial}, after={after}"
    );
}

// ===========================================================================
// 10. Learning rate change mid-training
// ===========================================================================

#[test]
fn test_lr_change_affects_step_size() {
    let w = var_from(vec![1.0], &[1]);
    let x = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    let target = DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap();

    let mut sgd = Sgd::new(vec![w.clone()], sgd_config(0.1, 0.0, 0.0)).unwrap();

    // Step with lr=0.1
    let loss = quadratic_loss(&w, &x, &target);
    let grads = backward(&loss).unwrap();
    let w_before = w.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    sgd.step(&grads).unwrap();
    let w_after_small = w.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let delta_small = (w_after_small - w_before).abs();

    // Reset w and step with lr=1.0
    w.set(&DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap())
        .unwrap();
    sgd.set_learning_rate(1.0).unwrap();

    let loss = quadratic_loss(&w, &x, &target);
    let grads = backward(&loss).unwrap();
    let w_before = w.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    sgd.step(&grads).unwrap();
    let w_after_large = w.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let delta_large = (w_after_large - w_before).abs();

    assert!(
        delta_large > delta_small * 5.0,
        "10x lr should give ~10x step: small={delta_small}, large={delta_large}"
    );
}

// ===========================================================================
// 11. Multi-variable model convergence
// ===========================================================================

#[test]
fn test_two_variable_model_converges() {
    // y = w1 * x + w2, target = 2*x + 1
    let w1 = var_from(vec![0.0], &[1]);
    let w2 = var_from(vec![0.0], &[1]);

    let mut adam = AdamW::new(vec![w1.clone(), w2.clone()], adam_config(0.05, 0.0)).unwrap();

    let xs = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    for _ in 0..100 {
        for &xi in &xs {
            let x_t = DynTensor::from_vec(vec![xi], &[1], &cpu()).unwrap();
            let target_val = 2.0 * xi + 1.0;
            let target_t = DynTensor::from_vec(vec![target_val], &[1], &cpu()).unwrap();

            let tw1 = tracked(&w1);
            let tw2 = tracked(&w2);
            let tx = Arc::new(TrackedTensor::from_tensor(x_t));
            let tt = Arc::new(TrackedTensor::from_tensor(target_t));

            let pred = tw1.mul(&tx).unwrap().add(&tw2).unwrap();
            let loss = pred.mse_loss(&tt).unwrap();
            let grads = backward(&loss).unwrap();
            adam.step(&grads).unwrap();
        }
    }

    let w1_val = w1.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let w2_val = w2.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    assert!(
        (w1_val - 2.0).abs() < 0.2,
        "w1 should converge near 2.0, got {w1_val}"
    );
    assert!(
        (w2_val - 1.0).abs() < 0.2,
        "w2 should converge near 1.0, got {w2_val}"
    );
}

// ===========================================================================
// 12. SGD with momentum converges faster than vanilla SGD
// ===========================================================================

#[test]
fn test_sgd_momentum_convergence() {
    let w_vanilla = var_from(vec![0.0, 0.0, 0.0], &[3]);
    let w_momentum = var_from(vec![0.0, 0.0, 0.0], &[3]);
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let target = DynTensor::from_vec(vec![2.0, 4.0, 6.0], &[3], &cpu()).unwrap();

    let mut sgd_v = Sgd::new(vec![w_vanilla.clone()], sgd_config(0.005, 0.0, 0.0)).unwrap();
    let mut sgd_m = Sgd::new(vec![w_momentum.clone()], sgd_config(0.005, 0.9, 0.0)).unwrap();

    for _ in 0..100 {
        let loss_v = quadratic_loss(&w_vanilla, &x, &target);
        let grads_v = backward(&loss_v).unwrap();
        sgd_v.step(&grads_v).unwrap();

        let loss_m = quadratic_loss(&w_momentum, &x, &target);
        let grads_m = backward(&loss_m).unwrap();
        sgd_m.step(&grads_m).unwrap();
    }

    let loss_vanilla = eval_quadratic_loss(&w_vanilla, &x, &target);
    let loss_momentum = eval_quadratic_loss(&w_momentum, &x, &target);

    assert!(
        loss_momentum < loss_vanilla,
        "momentum should converge faster: vanilla_loss={loss_vanilla}, momentum_loss={loss_momentum}"
    );
}

// ===========================================================================
// 13. Matmul-based linear layer: forward -> backward -> step
// ===========================================================================

#[test]
fn test_matmul_linear_layer_optimization() {
    // W: [2, 3], x: [1, 3], y = x @ W^T => [1, 2], target: [1, 2]
    let w = var_from(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let target = DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &cpu()).unwrap();

    let mut adam = AdamW::new(vec![w.clone()], adam_config(0.01, 0.0)).unwrap();

    let initial_loss = {
        let tw = tracked(&w);
        let tx = Arc::new(TrackedTensor::from_tensor(x.clone()));
        let tt = Arc::new(TrackedTensor::from_tensor(target.clone()));
        let wt = tw.transpose(0, 1).unwrap();
        let pred = tx.matmul(&wt).unwrap();
        let loss = pred.mse_loss(&tt).unwrap();
        get_scalar(loss.tensor())
    };

    for _ in 0..50 {
        let tw = tracked(&w);
        let tx = Arc::new(TrackedTensor::from_tensor(x.clone()));
        let tt = Arc::new(TrackedTensor::from_tensor(target.clone()));
        let wt = tw.transpose(0, 1).unwrap();
        let pred = tx.matmul(&wt).unwrap();
        let loss = pred.mse_loss(&tt).unwrap();
        let grads = backward(&loss).unwrap();
        adam.step(&grads).unwrap();
    }

    let final_loss = {
        let tw = tracked(&w);
        let tx = Arc::new(TrackedTensor::from_tensor(x));
        let tt = Arc::new(TrackedTensor::from_tensor(target));
        let wt = tw.transpose(0, 1).unwrap();
        let pred = tx.matmul(&wt).unwrap();
        let loss = pred.mse_loss(&tt).unwrap();
        get_scalar(loss.tensor())
    };

    assert!(
        final_loss < initial_loss * 0.1,
        "matmul model should reduce loss by >90%: initial={initial_loss}, final={final_loss}"
    );
}

// ===========================================================================
// 14. clip_grad_value integration
// ===========================================================================

#[test]
fn test_clip_grad_value_then_step() {
    let w = var_from(vec![50.0, -50.0], &[2]);
    let tw = tracked(&w);
    let sq = tw.sqr().unwrap();
    let loss = scalar_loss(&sq);
    let mut grads = backward(&loss).unwrap();

    // d/dw(w^2) = 2w => [100, -100]
    let grad_before = grads.get(&w).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(grad_before[0].abs() > 10.0);

    clip_grad_value(&mut grads, 5.0).unwrap();

    let grad_after = grads.get(&w).unwrap().to_flat_vec::<f32>().unwrap();
    for &g in &grad_after {
        assert!(
            g.abs() <= 5.0 + 1e-6,
            "gradient element {g} exceeds clip_value 5.0"
        );
    }

    let mut sgd = Sgd::new(vec![w], sgd_config(0.01, 0.0, 0.0)).unwrap();
    sgd.step(&grads).unwrap();
}
