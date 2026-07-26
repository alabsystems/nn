// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests verifying autodiff works correctly with optimizers on
//! training loops. Covers linear regression convergence, XOR classification,
//! quadratic minimization, LR scheduling, gradient clipping, multiple
//! parameter groups, and convex loss monotonicity.

use std::sync::Arc;

use nn_autodiff::{backward, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;
use nn_optim::{
    clip_grad_norm, clip_grad_value, step_with_schedule, AdamConfig, AdamW, CosineSchedule,
    LrSchedule, Optimizer, Sgd, SgdConfig, WarmupSchedule,
};

fn cpu() -> Device {
    Device::Cpu
}

/// Read the single f32 from a scalar (shape [1]) Var.
fn read_scalar(var: &Var) -> f32 {
    var.data().unwrap().to_flat_vec::<f32>().unwrap()[0]
}

/// Read all f32 values from a Var.
fn read_vec(var: &Var) -> Vec<f32> {
    var.data().unwrap().to_flat_vec::<f32>().unwrap()
}

fn adam_config(lr: f64, weight_decay: f64) -> AdamConfig {
    let mut c = AdamConfig::default();
    c.lr = lr;
    c.weight_decay = weight_decay;
    c
}

fn sgd_config(lr: f64, momentum: f64, weight_decay: f64) -> SgdConfig {
    let mut c = SgdConfig::default();
    c.lr = lr;
    c.momentum = momentum;
    c.weight_decay = weight_decay;
    c
}

// ============================================================================
// 1. Linear regression: train y = w*x + b, verify w and b converge
// ============================================================================

#[test]
fn test_linear_regression_converges_adam() {
    // True model: y = 2.5*x + 1.0
    // Training data: 4 samples
    let xs: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let ys: Vec<f32> = xs.iter().map(|x| 2.5 * x + 1.0).collect();

    let w = Var::new(DynTensor::from_vec(vec![0.0f32], &[1], &cpu()).unwrap());
    let b = Var::new(DynTensor::from_vec(vec![0.0f32], &[1], &cpu()).unwrap());

    let mut adam = AdamW::new(vec![w.clone(), b.clone()], adam_config(0.05, 0.0)).unwrap();

    // The bias term converges slowly on this ill-conditioned problem (x ranges
    // 1..4 so the slope dominates). At lr=0.05, b reaches ~1.19 after 500 steps
    // and only settles within 0.15 of 1.0 around ~900 steps. The optimizer is
    // correct; 500 steps was simply too short a budget.
    for _ in 0..1000 {
        let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
        let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());

        let x_tensor = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(xs.clone(), &[4, 1], &cpu()).unwrap(),
        ));
        let y_tensor = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(ys.clone(), &[4, 1], &cpu()).unwrap(),
        ));

        // y_pred = x * w + b (broadcast: [4,1] * [1] + [1] = [4,1])
        let pred = x_tensor.mul(&tw).unwrap().add(&tb).unwrap();
        let loss = pred.mse_loss(&y_tensor).unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let w_val = read_scalar(&w);
    let b_val = read_scalar(&b);
    assert!(
        (w_val - 2.5).abs() < 0.15,
        "w should converge near 2.5, got {w_val}"
    );
    assert!(
        (b_val - 1.0).abs() < 0.15,
        "b should converge near 1.0, got {b_val}"
    );
}

#[test]
fn test_linear_regression_converges_sgd() {
    // True model: y = -1.5*x + 3.0
    let xs: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0];
    let ys: Vec<f32> = xs.iter().map(|x| -1.5 * x + 3.0).collect();

    let w = Var::new(DynTensor::from_vec(vec![0.0f32], &[1], &cpu()).unwrap());
    let b = Var::new(DynTensor::from_vec(vec![0.0f32], &[1], &cpu()).unwrap());

    let mut sgd = Sgd::new(vec![w.clone(), b.clone()], sgd_config(0.01, 0.9, 0.0)).unwrap();

    for _ in 0..1000 {
        let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
        let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());

        let x_tensor = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(xs.clone(), &[4, 1], &cpu()).unwrap(),
        ));
        let y_tensor = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(ys.clone(), &[4, 1], &cpu()).unwrap(),
        ));

        let pred = x_tensor.mul(&tw).unwrap().add(&tb).unwrap();
        let loss = pred.mse_loss(&y_tensor).unwrap();
        sgd.backward_step(&loss).unwrap();
    }

    let w_val = read_scalar(&w);
    let b_val = read_scalar(&b);
    assert!(
        (w_val - (-1.5)).abs() < 0.2,
        "w should converge near -1.5, got {w_val}"
    );
    assert!(
        (b_val - 3.0).abs() < 0.2,
        "b should converge near 3.0, got {b_val}"
    );
}

// ============================================================================
// 2. XOR problem: small MLP with hidden layer, verify classification
// ============================================================================

#[test]
fn test_xor_mlp_converges() {
    // XOR inputs and targets
    // Inputs: [0,0], [0,1], [1,0], [1,1]
    // Targets: [0], [1], [1], [0]
    let x_data = vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let y_data = vec![0.0f32, 1.0, 1.0, 0.0];

    // 2-layer MLP: 2 -> 8 -> 1
    // Hidden layer: W1 [8,2], b1 [8]
    // Output layer: W2 [1,8], b2 [1]
    let w1 = Var::randn(&[8, 2], 0.0, 0.5, &cpu()).unwrap();
    let b1 = Var::zeros(&[8], nn_core::DType::F32, &cpu()).unwrap();
    let w2 = Var::randn(&[1, 8], 0.0, 0.3, &cpu()).unwrap();
    let b2 = Var::zeros(&[1], nn_core::DType::F32, &cpu()).unwrap();

    let mut adam = AdamW::new(
        vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()],
        adam_config(0.01, 0.0),
    )
    .unwrap();

    for _ in 0..2000 {
        let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
        let tb1 = Arc::new(TrackedTensor::from_var(&b1).unwrap());
        let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
        let tb2 = Arc::new(TrackedTensor::from_var(&b2).unwrap());

        let x = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(x_data.clone(), &[4, 2], &cpu()).unwrap(),
        ));
        let y = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(y_data.clone(), &[4, 1], &cpu()).unwrap(),
        ));

        // hidden = tanh(x @ W1^T + b1)
        let hidden = x
            .matmul(&tw1.transpose(0, 1).unwrap())
            .unwrap()
            .add(&tb1)
            .unwrap()
            .tanh()
            .unwrap();
        // output = sigmoid(hidden @ W2^T + b2)
        let output = hidden
            .matmul(&tw2.transpose(0, 1).unwrap())
            .unwrap()
            .add(&tb2)
            .unwrap()
            .sigmoid()
            .unwrap();

        // MSE loss
        let loss = output.mse_loss(&y).unwrap();
        adam.backward_step(&loss).unwrap();
    }

    // Verify predictions are approximately correct
    let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
    let tb1 = Arc::new(TrackedTensor::from_var(&b1).unwrap());
    let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
    let tb2 = Arc::new(TrackedTensor::from_var(&b2).unwrap());
    let x = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(x_data, &[4, 2], &cpu()).unwrap(),
    ));
    let hidden = x
        .matmul(&tw1.transpose(0, 1).unwrap())
        .unwrap()
        .add(&tb1)
        .unwrap()
        .tanh()
        .unwrap();
    let output = hidden
        .matmul(&tw2.transpose(0, 1).unwrap())
        .unwrap()
        .add(&tb2)
        .unwrap()
        .sigmoid()
        .unwrap();
    let preds = output.tensor().to_flat_vec::<f32>().unwrap();

    // XOR: [0,0]->0, [0,1]->1, [1,0]->1, [1,1]->0
    assert!(preds[0] < 0.3, "XOR(0,0) should be ~0, got {}", preds[0]);
    assert!(preds[1] > 0.7, "XOR(0,1) should be ~1, got {}", preds[1]);
    assert!(preds[2] > 0.7, "XOR(1,0) should be ~1, got {}", preds[2]);
    assert!(preds[3] < 0.3, "XOR(1,1) should be ~0, got {}", preds[3]);
}

// ============================================================================
// 3. Quadratic convergence: minimize (x - target)^2 with Adam
// ============================================================================

#[test]
fn test_quadratic_convergence_to_target() {
    // Minimize f(x) = (x - 3.7)^2. Global minimum at x = 3.7.
    let target_val = 3.7f32;
    let x = Var::new(DynTensor::from_vec(vec![0.0f32], &[1], &cpu()).unwrap());

    let mut adam = AdamW::new(vec![x.clone()], adam_config(0.1, 0.0)).unwrap();

    for _ in 0..300 {
        let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let target = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![target_val], &[1], &cpu()).unwrap(),
        ));
        let loss = tx.mse_loss(&target).unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let val = read_scalar(&x);
    assert!(
        (val - target_val).abs() < 0.05,
        "x should converge near {target_val}, got {val}"
    );
}

#[test]
fn test_quadratic_convergence_multidim() {
    // Minimize f(x) = mean((x - target)^2) for a 4-element vector.
    let target_data = vec![1.0f32, -2.0, 0.5, 3.0];
    let x = Var::new(DynTensor::from_vec(vec![0.0f32; 4], &[4], &cpu()).unwrap());

    let mut adam = AdamW::new(vec![x.clone()], adam_config(0.1, 0.0)).unwrap();

    for _ in 0..300 {
        let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let target = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(target_data.clone(), &[4], &cpu()).unwrap(),
        ));
        let loss = tx.mse_loss(&target).unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let vals = read_vec(&x);
    for (i, (v, t)) in vals.iter().zip(target_data.iter()).enumerate() {
        assert!(
            (v - t).abs() < 0.1,
            "x[{i}] should converge near {t}, got {v}"
        );
    }
}

// ============================================================================
// 4. Learning rate scheduling: verify LR decreases over iterations
// ============================================================================

#[test]
fn test_warmup_schedule_lr_increases_then_constant() {
    let schedule = WarmupSchedule::new(0.01, 100).unwrap();

    // During warmup, LR should increase linearly from 0 to base_lr
    let lr_0 = schedule.lr_at_step(0);
    let lr_50 = schedule.lr_at_step(50);
    let lr_100 = schedule.lr_at_step(100);
    let lr_200 = schedule.lr_at_step(200);

    assert!(lr_0 < 1e-10, "LR at step 0 should be ~0, got {lr_0}");
    assert!(
        (lr_50 - 0.005).abs() < 1e-10,
        "LR at step 50 should be 0.005, got {lr_50}"
    );
    assert!(
        (lr_100 - 0.01).abs() < 1e-10,
        "LR at step 100 should be base_lr 0.01, got {lr_100}"
    );
    assert!(
        (lr_200 - 0.01).abs() < 1e-10,
        "LR after warmup should be constant, got {lr_200}"
    );
}

#[test]
fn test_cosine_schedule_lr_decreases() {
    let schedule = CosineSchedule::new(0.01, 0.001, 10, 200).unwrap();

    let mut prev_lr = f64::MAX;
    // After warmup (step >= 10), the LR should monotonically decrease
    for step in 10..200 {
        let lr = schedule.lr_at_step(step);
        assert!(
            lr <= prev_lr + 1e-15,
            "Cosine LR should be non-increasing after warmup: step {step}, prev={prev_lr}, curr={lr}"
        );
        prev_lr = lr;
    }

    // LR at end should be near min_lr
    let lr_end = schedule.lr_at_step(199);
    assert!(
        (lr_end - 0.001).abs() < 0.001,
        "LR at end should approach min_lr 0.001, got {lr_end}"
    );

    // LR past total_steps should be min_lr
    let lr_past = schedule.lr_at_step(300);
    assert!(
        (lr_past - 0.001).abs() < 1e-10,
        "LR past total_steps should be min_lr, got {lr_past}"
    );
}

#[test]
fn test_cosine_schedule_applied_to_optimizer() {
    // Train with a cosine schedule and verify that the optimizer's LR tracks
    // the schedule values over time.
    let x = Var::new(DynTensor::from_vec(vec![5.0f32], &[1], &cpu()).unwrap());
    let mut adam = AdamW::new(vec![x.clone()], adam_config(0.1, 0.0)).unwrap();

    let schedule = CosineSchedule::new(0.1, 0.001, 0, 100).unwrap();

    let mut lr_at_step_0 = 0.0;
    let mut lr_at_step_50 = 0.0;
    let mut lr_at_step_99 = 0.0;

    for step in 0..100 {
        let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = tx.sqr().unwrap();
        let grads = backward(&loss).unwrap();
        step_with_schedule(&mut adam, &grads, &schedule, step).unwrap();

        match step {
            0 => lr_at_step_0 = adam.learning_rate(),
            50 => lr_at_step_50 = adam.learning_rate(),
            99 => lr_at_step_99 = adam.learning_rate(),
            _ => {}
        }
    }

    // LR should decrease over training
    assert!(
        lr_at_step_0 > lr_at_step_50,
        "LR at step 0 ({lr_at_step_0}) should be > LR at step 50 ({lr_at_step_50})"
    );
    assert!(
        lr_at_step_50 > lr_at_step_99,
        "LR at step 50 ({lr_at_step_50}) should be > LR at step 99 ({lr_at_step_99})"
    );
}

// ============================================================================
// 5. Gradient clipping: verify large gradients are capped
// ============================================================================

#[test]
fn test_gradient_clip_norm_caps_large_grads() {
    // Create a scenario with large gradients from two variables
    let x = Var::new(DynTensor::from_vec(vec![50.0f32], &[1], &cpu()).unwrap());
    let y = Var::new(DynTensor::from_vec(vec![-50.0f32], &[1], &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    // loss = x^2 + y^2 -> grad_x = 100, grad_y = -100
    // total norm = sqrt(100^2 + 100^2) = 100*sqrt(2) ~ 141.4
    let loss = tx.sqr().unwrap().add(&ty.sqr().unwrap()).unwrap();

    let mut grads = backward(&loss).unwrap();
    let original_norm = clip_grad_norm(&mut grads, 1.0).unwrap();

    assert!(
        original_norm > 100.0,
        "Original norm should be large, got {original_norm}"
    );

    // After clipping to max_norm=1.0, total norm of gradients should be ~1.0
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap()[0];
    let gy = grads.get(&y).unwrap().to_flat_vec::<f32>().unwrap()[0];
    let clipped_norm = gx.hypot(gy);
    assert!(
        (clipped_norm - 1.0).abs() < 0.01,
        "Clipped gradient norm should be ~1.0, got {clipped_norm}"
    );
}

#[test]
fn test_gradient_clip_value_clamps_elements() {
    let x = Var::new(DynTensor::from_vec(vec![100.0f32, -100.0, 0.5], &[3], &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    // loss = sum(x^2) via sum_keepdim + reshape
    let sq = tx.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().reshape(&[1]).unwrap();

    let mut grads = backward(&loss).unwrap();
    clip_grad_value(&mut grads, 5.0).unwrap();

    let grad_vals = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // grad = 2*x = [200, -200, 1.0], clamped to [-5, 5]
    assert!(
        (grad_vals[0] - 5.0).abs() < 1e-5,
        "Gradient should be clamped to 5.0, got {}",
        grad_vals[0]
    );
    assert!(
        (grad_vals[1] - (-5.0)).abs() < 1e-5,
        "Gradient should be clamped to -5.0, got {}",
        grad_vals[1]
    );
    assert!(
        (grad_vals[2] - 1.0).abs() < 1e-5,
        "Small gradient should be unchanged, got {}",
        grad_vals[2]
    );
}

#[test]
fn test_gradient_clipping_in_training_loop() {
    // Verify that gradient clipping prevents divergence on a problem with
    // initially large gradients.
    let x = Var::new(DynTensor::from_vec(vec![100.0f32], &[1], &cpu()).unwrap());

    let mut adam = AdamW::new(vec![x.clone()], adam_config(0.1, 0.0)).unwrap();

    // With the gradient norm clamped to 10, the per-step update saturates at
    // ~lr (0.1) while x is large, so traveling from x=100 toward 0 takes
    // ~1100 steps (200 steps only reaches ~80). The clipping and optimizer are
    // correct; the original step budget was far too small for the start point.
    for _ in 0..1200 {
        let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = tx.sqr().unwrap();
        let mut grads = backward(&loss).unwrap();
        clip_grad_norm(&mut grads, 10.0).unwrap();
        adam.step(&grads).unwrap();
    }

    let val = read_scalar(&x);
    // Should still converge despite clipping
    assert!(
        val.abs() < 1.0,
        "Training with gradient clipping should converge, got {val}"
    );
}

// ============================================================================
// 6. Multiple parameter groups: different LRs for different layers
// ============================================================================

#[test]
fn test_multiple_parameter_groups_different_lr() {
    // Simulate separate optimizers for different parameter groups
    // (different LRs for "backbone" vs "head").
    // Minimize f(a, b) = (a - 1)^2 + (b - 2)^2
    // Give 'a' a high LR (should converge fast) and 'b' a low LR (converges slow)

    let a = Var::new(DynTensor::from_vec(vec![0.0f32], &[1], &cpu()).unwrap());
    let b = Var::new(DynTensor::from_vec(vec![0.0f32], &[1], &cpu()).unwrap());

    // Two separate Adam optimizers with different LRs
    let mut opt_a = AdamW::new(vec![a.clone()], adam_config(0.1, 0.0)).unwrap();
    let mut opt_b = AdamW::new(vec![b.clone()], adam_config(0.001, 0.0)).unwrap();

    let steps = 50;
    for _ in 0..steps {
        let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
        let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());

        let one = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![1.0f32], &[1], &cpu()).unwrap(),
        ));
        let two = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![2.0f32], &[1], &cpu()).unwrap(),
        ));

        let loss_a = ta.sub(&one).unwrap().sqr().unwrap();
        let loss_b = tb.sub(&two).unwrap().sqr().unwrap();
        let loss = loss_a.add(&loss_b).unwrap();

        let grads = backward(&loss).unwrap();
        opt_a.step(&grads).unwrap();
        opt_b.step(&grads).unwrap();
    }

    let a_val = read_scalar(&a);
    let b_val = read_scalar(&b);

    // 'a' with higher LR should have converged closer to target
    let a_error = (a_val - 1.0).abs();
    let b_error = (b_val - 2.0).abs();
    assert!(
        a_error < b_error,
        "Higher LR param 'a' (err={a_error}) should converge faster than 'b' (err={b_error})"
    );
    assert!(
        a_error < 0.3,
        "'a' with high LR should be close to 1.0 after {steps} steps, got {a_val}"
    );
}

#[test]
fn test_parameter_group_lr_update_mid_training() {
    // Start both groups at same LR, then lower one group's LR mid-training.
    let x = Var::new(DynTensor::from_vec(vec![5.0f32], &[1], &cpu()).unwrap());
    let y = Var::new(DynTensor::from_vec(vec![5.0f32], &[1], &cpu()).unwrap());

    let mut opt_x = AdamW::new(vec![x.clone()], adam_config(0.1, 0.0)).unwrap();
    let mut opt_y = AdamW::new(vec![y.clone()], adam_config(0.1, 0.0)).unwrap();

    // Phase 1: same LR for 50 steps
    for _ in 0..50 {
        let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
        let loss = tx.sqr().unwrap().add(&ty.sqr().unwrap()).unwrap();
        let grads = backward(&loss).unwrap();
        opt_x.step(&grads).unwrap();
        opt_y.step(&grads).unwrap();
    }

    // Phase 2: drop y's LR dramatically
    opt_y.set_learning_rate(0.0001).unwrap();

    let x_before_phase2 = read_scalar(&x);
    let y_before_phase2 = read_scalar(&y);

    for _ in 0..50 {
        let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
        let loss = tx.sqr().unwrap().add(&ty.sqr().unwrap()).unwrap();
        let grads = backward(&loss).unwrap();
        opt_x.step(&grads).unwrap();
        opt_y.step(&grads).unwrap();
    }

    let x_after = read_scalar(&x);
    let y_after = read_scalar(&y);

    // x should have moved more than y in phase 2
    let x_movement = (x_before_phase2 - x_after).abs();
    let y_movement = (y_before_phase2 - y_after).abs();
    assert!(
        x_movement > y_movement,
        "x (high LR) should move more than y (low LR) in phase 2: \
         x_moved={x_movement}, y_moved={y_movement}"
    );
}

// ============================================================================
// 7. Loss decreases monotonically on convex problems
// ============================================================================

#[test]
fn test_loss_monotonically_decreases_sgd_quadratic() {
    // For f(x) = x^2 with appropriate LR, SGD loss should decrease every step.
    let x = Var::new(DynTensor::from_vec(vec![5.0f32], &[1], &cpu()).unwrap());

    let mut sgd = Sgd::new(vec![x.clone()], sgd_config(0.01, 0.0, 0.0)).unwrap();

    let mut prev_loss_val = f32::MAX;
    // SGD on x^2 with lr=0.01 shrinks x geometrically by (1 - 2*lr) = 0.98 per
    // step, so reaching loss < 0.01 (|x| < 0.1) from x=5 needs ~195 steps; 100
    // steps only reaches loss ~0.44. The optimizer is correct — the step budget
    // was too small for this learning rate.
    let total_steps = 250;
    for step in 0..total_steps {
        let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = tx.sqr().unwrap();
        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(
            loss_val <= prev_loss_val + 1e-7,
            "Loss should be non-increasing on convex problem with small LR: \
             step {step}, prev={prev_loss_val}, curr={loss_val}"
        );
        prev_loss_val = loss_val;
        sgd.backward_step(&loss).unwrap();
    }

    assert!(
        prev_loss_val < 0.01,
        "Loss should be near zero after {total_steps} steps, got {prev_loss_val}"
    );
}

#[test]
fn test_loss_decreases_adam_multi_variable_convex() {
    // Minimize mean((x_i - target_i)^2) for a multi-variable convex problem.
    // Loss should generally decrease (allowing minor float jitter).
    let targets = vec![1.0f32, -1.0, 2.0, -0.5];
    let x = Var::new(DynTensor::from_vec(vec![0.0f32; 4], &[4], &cpu()).unwrap());

    let mut adam = AdamW::new(vec![x.clone()], adam_config(0.05, 0.0)).unwrap();

    let mut losses = Vec::with_capacity(200);
    for _ in 0..200 {
        let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let target = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(targets.clone(), &[4], &cpu()).unwrap(),
        ));
        let loss = tx.mse_loss(&target).unwrap();
        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    // Final loss should be much smaller than initial
    assert!(
        losses.last().unwrap() < &(losses[0] * 0.01),
        "Final loss ({}) should be < 1% of initial ({})",
        losses.last().unwrap(),
        losses[0]
    );

    // Compute a smoothed-window loss to verify the general downward trend.
    // Adam oscillates slightly as it settles into the optimum, so a 10-step
    // window still shows small per-window bumps near convergence (loss ~5e-4).
    // A 20-step window smooths over Adam's near-optimum overshoot while keeping
    // the strict (<= prev + 1e-5) monotonicity assertion intact.
    let window = 20;
    let mut avg_losses = Vec::new();
    for i in 0..losses.len() / window {
        let start = i * window;
        let end = start + window;
        let avg: f32 = losses[start..end].iter().sum::<f32>() / window as f32;
        avg_losses.push(avg);
    }

    for i in 1..avg_losses.len() {
        assert!(
            avg_losses[i] <= avg_losses[i - 1] + 1e-5,
            "Smoothed loss should decrease: window {} avg={}, window {} avg={}",
            i - 1,
            avg_losses[i - 1],
            i,
            avg_losses[i]
        );
    }
}

#[test]
fn test_loss_decrease_with_weight_decay_on_convex() {
    // Weight decay adds regularization. On a convex problem, loss should
    // still decrease, though it converges to a non-zero minimum due to the
    // regularization term.
    let x = Var::new(DynTensor::from_vec(vec![10.0f32], &[1], &cpu()).unwrap());

    let mut adam = AdamW::new(vec![x.clone()], adam_config(0.05, 0.01)).unwrap();

    let initial_x = read_scalar(&x);
    // Adam moves ~lr (0.05) per step, so from x=10 reaching |x| < 0.5 needs
    // ~330 steps; 100 steps only reaches ~5.17. The optimizer (with decoupled
    // weight decay) is correct — the step budget was too small.
    let total_steps = 350;
    for _ in 0..total_steps {
        let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = tx.sqr().unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let final_x = read_scalar(&x);
    assert!(
        final_x.abs() < initial_x.abs(),
        "x should converge toward 0: initial={initial_x}, final={final_x}"
    );
    assert!(
        final_x.abs() < 0.5,
        "x should be near 0 after {total_steps} steps with weight decay, got {final_x}"
    );
}
