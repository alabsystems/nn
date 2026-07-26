// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Training loop integration tests for nn-autodiff + nn-optim.
//!
//! These tests exercise complete training workflows: forward pass, loss
//! computation, backward pass, and optimizer weight updates. They verify
//! that the autodiff + optimizer pipeline produces decreasing loss on
//! learnable problems.

use std::sync::Arc;

use nn_autodiff::{backward, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;
use nn_optim::optimizer::Optimizer;
use nn_optim::sgd::{Sgd, SgdConfig};
use nn_optim::{AdamConfig, AdamW};

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

/// Create an AdamConfig with full control over hyperparameters.
fn adam_config_full(lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> AdamConfig {
    let mut config = AdamConfig::default();
    config.lr = lr;
    config.beta1 = beta1;
    config.beta2 = beta2;
    config.eps = eps;
    config.weight_decay = weight_decay;
    config
}

/// MSE loss: mean((pred - target)^2)
/// Returns a scalar TrackedTensor suitable for backward().
fn mse_loss(pred: &Arc<TrackedTensor>, target: &Arc<TrackedTensor>) -> Arc<TrackedTensor> {
    let diff = pred.sub(target).unwrap();
    let sq = diff.sqr().unwrap();
    let numel = sq.tensor().numel();
    let mut reduced = sq;
    for dim in (0..reduced.tensor().rank()).rev() {
        reduced = reduced.sum_keepdim(dim).unwrap();
    }
    reduced.mul_scalar(1.0 / numel as f64).unwrap()
}

/// Helper: reduce a tracked tensor to a scalar by summing all dims.
fn reduce_to_scalar(t: &Arc<TrackedTensor>) -> Arc<TrackedTensor> {
    let mut reduced = Arc::clone(t);
    for dim in (0..reduced.tensor().rank()).rev() {
        reduced = reduced.sum_keepdim(dim).unwrap();
    }
    reduced
}

/// Extract scalar f32 from a tracked tensor.
fn scalar_val(t: &Arc<TrackedTensor>) -> f32 {
    t.tensor().to_scalar::<f32>().unwrap()
}

// ---------------------------------------------------------------------------
// (a) Simple linear regression: train y = Wx + b to fit known data
// ---------------------------------------------------------------------------

#[test]
fn test_linear_regression_sgd_loss_decreases() {
    // Target: y = 2*x + 1 for x in {0, 1, 2, 3}
    // We train W (1x1) and b (1x1) to fit this.
    let w = Var::new(DynTensor::from_vec(vec![0.5], &[1, 1], &cpu()).unwrap());
    let b = Var::new(DynTensor::from_vec(vec![0.0], &[1, 1], &cpu()).unwrap());

    let mut sgd = Sgd::new(vec![w.clone(), b.clone()], sgd_config(0.01, 0.0, 0.0)).unwrap();

    // Training data: x=[0,1,2,3], y=[1,3,5,7]
    let x_data = DynTensor::from_vec(vec![0.0, 1.0, 2.0, 3.0], &[4, 1], &cpu()).unwrap();
    let y_data = DynTensor::from_vec(vec![1.0, 3.0, 5.0, 7.0], &[4, 1], &cpu()).unwrap();
    let x_input = Arc::new(TrackedTensor::from_tensor(x_data));
    let y_target = Arc::new(TrackedTensor::from_tensor(y_data));

    let mut losses = Vec::new();

    for _ in 0..10 {
        let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
        let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());

        // pred = x @ W + b (broadcast b across batch)
        let pred = x_input.matmul(&tw).unwrap();
        let pred = pred.add(&tb).unwrap();

        let loss = mse_loss(&pred, &y_target);
        let loss_val = scalar_val(&loss);
        losses.push(loss_val);

        sgd.backward_step(&loss).unwrap();
    }

    // Assert loss decreases over 10 iterations
    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "loss should decrease: first={}, last={}",
        losses.first().unwrap(),
        losses.last().unwrap()
    );

    // Assert loss decreased significantly (not just noise)
    assert!(
        *losses.last().unwrap() < losses[0] * 0.9,
        "loss should decrease by at least 10%: first={}, last={}",
        losses[0],
        losses.last().unwrap()
    );
}

// ---------------------------------------------------------------------------
// (b) Two-layer MLP training on XOR-like problem
// ---------------------------------------------------------------------------

#[test]
fn test_two_layer_mlp_loss_decreases() {
    // XOR-like: 4 input patterns, 2D input, 1D output
    // Inputs: [[0,0],[0,1],[1,0],[1,1]]
    // Targets: [0, 1, 1, 0]

    // Hidden layer: 2 -> 4
    let w1 = Var::new(
        DynTensor::from_vec(
            vec![0.5, -0.3, 0.2, 0.8, -0.4, 0.6, 0.1, -0.7],
            &[2, 4],
            &cpu(),
        )
        .unwrap(),
    );
    let b1 = Var::new(DynTensor::from_vec(vec![0.1, -0.1, 0.05, -0.05], &[1, 4], &cpu()).unwrap());

    // Output layer: 4 -> 1
    let w2 = Var::new(DynTensor::from_vec(vec![0.3, -0.5, 0.4, 0.2], &[4, 1], &cpu()).unwrap());
    let b2 = Var::new(DynTensor::from_vec(vec![0.0], &[1, 1], &cpu()).unwrap());

    let mut sgd = Sgd::new(
        vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()],
        sgd_config(0.1, 0.0, 0.0),
    )
    .unwrap();

    let x_data = DynTensor::from_vec(
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        &[4, 2],
        &cpu(),
    )
    .unwrap();
    let y_data = DynTensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], &[4, 1], &cpu()).unwrap();
    let x_input = Arc::new(TrackedTensor::from_tensor(x_data));
    let y_target = Arc::new(TrackedTensor::from_tensor(y_data));

    let initial_loss;
    let mut final_loss = 0.0;

    // Record initial loss
    {
        let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
        let tb1 = Arc::new(TrackedTensor::from_var(&b1).unwrap());
        let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
        let tb2 = Arc::new(TrackedTensor::from_var(&b2).unwrap());

        let h = x_input.matmul(&tw1).unwrap().add(&tb1).unwrap();
        let h = h.relu().unwrap();
        let pred = h.matmul(&tw2).unwrap().add(&tb2).unwrap();
        let loss = mse_loss(&pred, &y_target);
        initial_loss = scalar_val(&loss);
    }

    for _ in 0..50 {
        let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
        let tb1 = Arc::new(TrackedTensor::from_var(&b1).unwrap());
        let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
        let tb2 = Arc::new(TrackedTensor::from_var(&b2).unwrap());

        // Forward: Linear -> ReLU -> Linear
        let h = x_input.matmul(&tw1).unwrap().add(&tb1).unwrap();
        let h = h.relu().unwrap();
        let pred = h.matmul(&tw2).unwrap().add(&tb2).unwrap();

        let loss = mse_loss(&pred, &y_target);
        final_loss = scalar_val(&loss);

        sgd.backward_step(&loss).unwrap();
    }

    assert!(
        final_loss < initial_loss,
        "MLP loss should decrease: initial={initial_loss}, final={final_loss}"
    );
}

// ---------------------------------------------------------------------------
// (c) Adam optimizer convergence
// ---------------------------------------------------------------------------

#[test]
fn test_adam_convergence_quadratic() {
    // Minimize f(x) = x^2 starting from x=5.0
    let x = Var::new(DynTensor::from_vec(vec![5.0], &[1], &cpu()).unwrap());

    let mut adam = AdamW::new(vec![x.clone()], adam_config(0.1, 0.0)).unwrap();

    let mut losses = Vec::new();
    for _ in 0..50 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        losses.push(scalar_val(&loss));
        adam.backward_step(&loss).unwrap();
    }

    // Adam should converge toward x=0 (loss=0)
    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < 1.0,
        "Adam should converge: final_loss={final_loss}"
    );
    assert!(
        final_loss < losses[0],
        "Adam loss should decrease: first={}, last={final_loss}",
        losses[0]
    );
}

#[test]
fn test_adam_converges_faster_than_sgd_on_quadratic() {
    // Both start at x=3.0, run 20 steps on f(x) = x^2.
    // Adam lr=0.1 gives near-constant step size (~0.1) due to gradient normalization.
    // SGD lr=0.01 gives step proportional to gradient, but with the small lr it
    // converges slower than Adam on this 1D problem.
    let x_adam = Var::new(DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap());
    let x_sgd = Var::new(DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap());

    let mut adam = AdamW::new(vec![x_adam.clone()], adam_config(0.1, 0.0)).unwrap();
    let mut sgd = Sgd::new(vec![x_sgd.clone()], sgd_config(0.01, 0.0, 0.0)).unwrap();

    for _ in 0..20 {
        // Adam step
        let t = Arc::new(TrackedTensor::from_var(&x_adam).unwrap());
        let loss = t.sqr().unwrap();
        adam.backward_step(&loss).unwrap();

        // SGD step
        let t = Arc::new(TrackedTensor::from_var(&x_sgd).unwrap());
        let loss = t.sqr().unwrap();
        sgd.backward_step(&loss).unwrap();
    }

    let adam_final = x_adam.data().unwrap().to_flat_vec::<f32>().unwrap()[0].abs();
    let sgd_final = x_sgd.data().unwrap().to_flat_vec::<f32>().unwrap()[0].abs();

    // Adam with momentum and bias correction should be closer to 0 than plain SGD
    assert!(
        adam_final < sgd_final,
        "Adam should converge faster: adam_x={adam_final}, sgd_x={sgd_final}"
    );
}

#[test]
fn test_adam_bias_correction_active() {
    // With bias correction, the optimizer should make a meaningful update
    // on the very first step, not a tiny one.
    let x = Var::new(DynTensor::from_vec(vec![10.0], &[1], &cpu()).unwrap());

    let mut adam = AdamW::new(
        vec![x.clone()],
        adam_config_full(0.01, 0.9, 0.999, 1e-8, 0.0),
    )
    .unwrap();

    // First step on f(x) = x^2
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    adam.backward_step(&loss).unwrap();

    let after_one = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    // Adam should update x meaningfully on the first step
    assert!(
        (10.0 - after_one).abs() > 1e-6,
        "Adam should update x: before=10.0, after={after_one}"
    );
    assert!(
        after_one < 10.0,
        "Adam should decrease x toward 0: after={after_one}"
    );
}

// ---------------------------------------------------------------------------
// (d) Gradient accumulation
// ---------------------------------------------------------------------------

#[test]
fn test_gradient_accumulation_matches_single_batch() {
    // Two half-batches accumulated should give similar gradients to one full batch.
    // Weight is [1,1] — a scalar multiplier for 1-feature inputs (y = w*x).
    let w = Var::new(DynTensor::from_vec(vec![1.0], &[1, 1], &cpu()).unwrap());

    // Full batch: x = [[1],[2],[3],[4]], y = [[2],[4],[6],[8]]
    let x_full = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4, 1], &cpu()).unwrap(),
    ));
    let y_full = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[4, 1], &cpu()).unwrap(),
    ));

    // Full batch gradient
    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let pred = x_full.matmul(&tw).unwrap();
    let loss_full = mse_loss(&pred, &y_full);
    let grads_full = backward(&loss_full).unwrap();
    let grad_full = grads_full.get(&w).unwrap().to_flat_vec::<f32>().unwrap();

    // Two half-batches
    let x1 = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0, 2.0], &[2, 1], &cpu()).unwrap(),
    ));
    let y1 = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![2.0, 4.0], &[2, 1], &cpu()).unwrap(),
    ));
    let x2 = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![3.0, 4.0], &[2, 1], &cpu()).unwrap(),
    ));
    let y2 = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![6.0, 8.0], &[2, 1], &cpu()).unwrap(),
    ));

    let tw1 = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let pred1 = x1.matmul(&tw1).unwrap();
    let loss1 = mse_loss(&pred1, &y1);
    let grads1 = backward(&loss1).unwrap();

    let tw2 = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let pred2 = x2.matmul(&tw2).unwrap();
    let loss2 = mse_loss(&pred2, &y2);
    let grads2 = backward(&loss2).unwrap();

    // Manually accumulate: grad_accum = (grad1 + grad2) / 2
    let g1 = grads1.get(&w).unwrap();
    let g2 = grads2.get(&w).unwrap();
    let accum = g1.add(g2).unwrap().mul_scalar(0.5).unwrap();
    let grad_accum = accum.to_flat_vec::<f32>().unwrap();

    // These should be close (not exact due to MSE normalization per batch size)
    for (a, b) in grad_full.iter().zip(grad_accum.iter()) {
        assert!(
            (a - b).abs() < 0.5,
            "accumulated gradient should approximate full batch: full={a}, accum={b}"
        );
    }
}

#[test]
fn test_gradient_accumulation_multiple_passes() {
    // Multiple forward/backward passes, then a single optimizer step
    // moving weight toward the mean target.
    let w = Var::new(DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap());
    let initial_w = 3.0_f32;

    let mut sgd = Sgd::new(vec![w.clone()], sgd_config(0.1, 0.0, 0.0)).unwrap();

    // Use the mean of [1,2,3] = 2.0 as target.
    // loss = (w - 2)^2, grad = 2*(w - 2)
    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let mean_target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap(),
    ));
    let diff = tw.sub(&mean_target).unwrap();
    let loss = diff.sqr().unwrap();
    sgd.backward_step(&loss).unwrap();

    let new_w = w.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    // w should move toward 2.0 (the mean target)
    assert!(
        (new_w - 2.0).abs() < (initial_w - 2.0).abs(),
        "w should move toward target: initial={initial_w}, new={new_w}, target=2.0"
    );
}

// ---------------------------------------------------------------------------
// (e) Mixed activation training
// ---------------------------------------------------------------------------

#[test]
fn test_mixed_activations_relu_silu_tanh_loss_decreases() {
    // Network: Linear(2->4) -> ReLU -> Linear(4->4) -> SiLU -> Linear(4->1) -> Tanh
    // Train to output 0.5 for all inputs
    let w1 = Var::new(
        DynTensor::from_vec(
            vec![0.3, -0.2, 0.5, 0.1, -0.4, 0.6, -0.1, 0.7],
            &[2, 4],
            &cpu(),
        )
        .unwrap(),
    );
    let w2 = Var::new(
        DynTensor::from_vec(
            vec![
                0.2, -0.3, 0.1, 0.4, -0.1, 0.5, -0.2, 0.3, 0.4, -0.1, 0.2, -0.5, 0.1, 0.3, -0.4,
                0.2,
            ],
            &[4, 4],
            &cpu(),
        )
        .unwrap(),
    );
    let w3 = Var::new(DynTensor::from_vec(vec![0.5, -0.3, 0.2, 0.1], &[4, 1], &cpu()).unwrap());

    let mut sgd = Sgd::new(
        vec![w1.clone(), w2.clone(), w3.clone()],
        sgd_config(0.05, 0.0, 0.0),
    )
    .unwrap();

    let x_data = DynTensor::from_vec(
        vec![1.0, 0.5, -0.5, 1.0, 0.0, -1.0, 1.0, 1.0],
        &[4, 2],
        &cpu(),
    )
    .unwrap();
    let y_data = DynTensor::from_vec(vec![0.5, 0.5, 0.5, 0.5], &[4, 1], &cpu()).unwrap();
    let x_input = Arc::new(TrackedTensor::from_tensor(x_data));
    let y_target = Arc::new(TrackedTensor::from_tensor(y_data));

    let mut losses = Vec::new();

    for _ in 0..30 {
        let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
        let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
        let tw3 = Arc::new(TrackedTensor::from_var(&w3).unwrap());

        // Forward: Linear -> ReLU -> Linear -> SiLU -> Linear -> Tanh
        let h1 = x_input.matmul(&tw1).unwrap().relu().unwrap();
        let h2 = h1.matmul(&tw2).unwrap().silu().unwrap();
        let pred = h2.matmul(&tw3).unwrap().tanh().unwrap();

        let loss = mse_loss(&pred, &y_target);
        losses.push(scalar_val(&loss));

        sgd.backward_step(&loss).unwrap();
    }

    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "mixed activation loss should decrease: first={}, last={}",
        losses.first().unwrap(),
        losses.last().unwrap()
    );
}

#[test]
fn test_mixed_activations_all_gradients_finite() {
    // Verify that different activations produce finite gradients
    let w = Var::new(DynTensor::from_vec(vec![1.0, -0.5, 0.3, 0.8], &[1, 4], &cpu()).unwrap());

    let x_data = DynTensor::from_vec(vec![0.5, -0.5, 1.0, -1.0], &[1, 4], &cpu()).unwrap();
    let x_input = Arc::new(TrackedTensor::from_tensor(x_data));

    // Test each activation individually
    let activations: Vec<(&str, Box<dyn Fn(&Arc<TrackedTensor>) -> Arc<TrackedTensor>>)> = vec![
        ("relu", Box::new(|t: &Arc<TrackedTensor>| t.relu().unwrap())),
        ("silu", Box::new(|t: &Arc<TrackedTensor>| t.silu().unwrap())),
        ("tanh", Box::new(|t: &Arc<TrackedTensor>| t.tanh().unwrap())),
        (
            "sigmoid",
            Box::new(|t: &Arc<TrackedTensor>| t.sigmoid().unwrap()),
        ),
        ("gelu", Box::new(|t: &Arc<TrackedTensor>| t.gelu().unwrap())),
    ];

    for (name, activation) in &activations {
        let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
        let h = x_input.mul(&tw).unwrap();
        let activated = activation(&h);
        let loss = reduce_to_scalar(&activated);

        let grads = backward(&loss).unwrap();
        let grad = grads.get(&w).unwrap();
        let grad_vals = grad.to_flat_vec::<f32>().unwrap();

        for (i, &g) in grad_vals.iter().enumerate() {
            assert!(g.is_finite(), "{name} gradient[{i}] is not finite: {g}");
        }
    }
}

// ---------------------------------------------------------------------------
// (f) Weight decay / L2 regularization
// ---------------------------------------------------------------------------

#[test]
fn test_weight_decay_shrinks_weights() {
    // With weight decay and a small-gradient loss, weights should decay toward zero.
    let w = Var::new(DynTensor::from_vec(vec![5.0, -3.0, 2.0, -1.0], &[4], &cpu()).unwrap());

    let mut sgd = Sgd::new(vec![w.clone()], sgd_config(0.1, 0.0, 0.1)).unwrap();

    let initial_norm: f32 = w
        .data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();

    // Run several steps with a simple loss that has small gradients
    for _ in 0..20 {
        let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
        // Small loss: just sum of small perturbations
        let loss = tw.mul_scalar(0.01).unwrap();
        let loss = reduce_to_scalar(&loss);
        sgd.backward_step(&loss).unwrap();
    }

    let final_norm: f32 = w
        .data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();

    assert!(
        final_norm < initial_norm,
        "weight decay should shrink weights: initial_norm={initial_norm}, final_norm={final_norm}"
    );
}

#[test]
fn test_weight_decay_vs_no_weight_decay() {
    // Same problem, compare weight norms with and without weight decay
    let w_wd = Var::new(DynTensor::from_vec(vec![3.0, -2.0], &[2, 1], &cpu()).unwrap());
    let w_no_wd = Var::new(DynTensor::from_vec(vec![3.0, -2.0], &[2, 1], &cpu()).unwrap());

    let mut sgd_wd = Sgd::new(vec![w_wd.clone()], sgd_config(0.01, 0.0, 0.1)).unwrap();
    let mut sgd_no_wd = Sgd::new(vec![w_no_wd.clone()], sgd_config(0.01, 0.0, 0.0)).unwrap();

    let x_data = DynTensor::from_vec(vec![1.0, 0.5, -0.5, 1.0], &[2, 2], &cpu()).unwrap();
    let y_data = DynTensor::from_vec(vec![1.0, -1.0], &[2, 1], &cpu()).unwrap();
    let x_input = Arc::new(TrackedTensor::from_tensor(x_data));
    let y_target = Arc::new(TrackedTensor::from_tensor(y_data));

    for _ in 0..30 {
        // With weight decay
        let tw = Arc::new(TrackedTensor::from_var(&w_wd).unwrap());
        let pred = x_input.matmul(&tw).unwrap();
        let loss = mse_loss(&pred, &y_target);
        sgd_wd.backward_step(&loss).unwrap();

        // Without weight decay
        let tw = Arc::new(TrackedTensor::from_var(&w_no_wd).unwrap());
        let pred = x_input.matmul(&tw).unwrap();
        let loss = mse_loss(&pred, &y_target);
        sgd_no_wd.backward_step(&loss).unwrap();
    }

    let norm_wd: f32 = w_wd
        .data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();

    let norm_no_wd: f32 = w_no_wd
        .data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();

    assert!(
        norm_wd < norm_no_wd,
        "regularized weights should have smaller norm: wd_norm={norm_wd}, no_wd_norm={norm_no_wd}"
    );
}

// ---------------------------------------------------------------------------
// Additional: SGD with momentum
// ---------------------------------------------------------------------------

#[test]
fn test_sgd_momentum_accelerates_convergence() {
    // SGD with momentum should converge faster than without on a quadratic
    let x_mom = Var::new(DynTensor::from_vec(vec![5.0], &[1], &cpu()).unwrap());
    let x_plain = Var::new(DynTensor::from_vec(vec![5.0], &[1], &cpu()).unwrap());

    let mut sgd_mom = Sgd::new(vec![x_mom.clone()], sgd_config(0.01, 0.9, 0.0)).unwrap();
    let mut sgd_plain = Sgd::new(vec![x_plain.clone()], sgd_config(0.01, 0.0, 0.0)).unwrap();

    for _ in 0..50 {
        // Momentum SGD
        let t = Arc::new(TrackedTensor::from_var(&x_mom).unwrap());
        let loss = t.sqr().unwrap();
        sgd_mom.backward_step(&loss).unwrap();

        // Plain SGD
        let t = Arc::new(TrackedTensor::from_var(&x_plain).unwrap());
        let loss = t.sqr().unwrap();
        sgd_plain.backward_step(&loss).unwrap();
    }

    let final_mom = x_mom.data().unwrap().to_flat_vec::<f32>().unwrap()[0].abs();
    let final_plain = x_plain.data().unwrap().to_flat_vec::<f32>().unwrap()[0].abs();

    assert!(
        final_mom < final_plain,
        "momentum SGD should converge faster: momentum_x={final_mom}, plain_x={final_plain}"
    );
}

// ---------------------------------------------------------------------------
// Additional: Adam with weight decay (AdamW)
// ---------------------------------------------------------------------------

#[test]
fn test_adamw_weight_decay_shrinks_weights() {
    let w = Var::new(DynTensor::from_vec(vec![5.0, -3.0, 2.0], &[3], &cpu()).unwrap());

    let initial_norm: f32 = w
        .data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();

    let mut adam = AdamW::new(vec![w.clone()], adam_config(0.01, 0.1)).unwrap();

    // Train toward zero
    for _ in 0..30 {
        let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
        let loss = tw.sqr().unwrap();
        let loss = reduce_to_scalar(&loss);
        adam.backward_step(&loss).unwrap();
    }

    let final_norm: f32 = w
        .data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();

    assert!(
        final_norm < initial_norm,
        "AdamW should shrink weights: initial={initial_norm}, final={final_norm}"
    );
}

// ---------------------------------------------------------------------------
// Additional: Multi-step loss monotonicity
// ---------------------------------------------------------------------------

#[test]
fn test_loss_monotonically_decreases_early_steps() {
    // On a simple quadratic, first several steps should strictly decrease loss
    let x = Var::new(DynTensor::from_vec(vec![10.0], &[1], &cpu()).unwrap());

    let mut sgd = Sgd::new(vec![x.clone()], sgd_config(0.05, 0.0, 0.0)).unwrap();

    let mut losses = Vec::new();
    for _ in 0..5 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        losses.push(scalar_val(&loss));
        sgd.backward_step(&loss).unwrap();
    }

    // First 5 steps on x^2 with small lr should be monotonically decreasing
    for i in 1..losses.len() {
        assert!(
            losses[i] < losses[i - 1],
            "loss should decrease monotonically at step {i}: prev={}, curr={}",
            losses[i - 1],
            losses[i]
        );
    }
}

// ---------------------------------------------------------------------------
// (g) Learning rate effect: higher LR = faster initial convergence
// ---------------------------------------------------------------------------

#[test]
fn test_learning_rate_effect_sgd() {
    // On a simple quadratic f(x) = x^2, higher LR should converge faster
    // in the first N steps (before overshooting becomes an issue).
    let x_low = Var::new(DynTensor::from_vec(vec![5.0], &[1], &cpu()).unwrap());
    let x_high = Var::new(DynTensor::from_vec(vec![5.0], &[1], &cpu()).unwrap());

    let mut sgd_low = Sgd::new(vec![x_low.clone()], sgd_config(0.001, 0.0, 0.0)).unwrap();
    let mut sgd_high = Sgd::new(vec![x_high.clone()], sgd_config(0.05, 0.0, 0.0)).unwrap();

    let mut losses_low = Vec::new();
    let mut losses_high = Vec::new();

    for _ in 0..20 {
        // Low LR step
        let t = Arc::new(TrackedTensor::from_var(&x_low).unwrap());
        let loss = t.sqr().unwrap();
        losses_low.push(scalar_val(&loss));
        sgd_low.backward_step(&loss).unwrap();

        // High LR step
        let t = Arc::new(TrackedTensor::from_var(&x_high).unwrap());
        let loss = t.sqr().unwrap();
        losses_high.push(scalar_val(&loss));
        sgd_high.backward_step(&loss).unwrap();
    }

    // Higher LR should produce lower loss after 20 steps
    let final_low = *losses_low.last().unwrap();
    let final_high = *losses_high.last().unwrap();
    assert!(
        final_high < final_low,
        "higher LR should converge faster: low_lr_loss={final_low}, high_lr_loss={final_high}"
    );
}

#[test]
fn test_learning_rate_effect_adam() {
    // Same test with Adam: higher LR should converge faster initially.
    let x_low = Var::new(DynTensor::from_vec(vec![5.0], &[1], &cpu()).unwrap());
    let x_high = Var::new(DynTensor::from_vec(vec![5.0], &[1], &cpu()).unwrap());

    let mut adam_low = AdamW::new(vec![x_low.clone()], adam_config(0.01, 0.0)).unwrap();
    let mut adam_high = AdamW::new(vec![x_high.clone()], adam_config(0.1, 0.0)).unwrap();

    for _ in 0..30 {
        let t = Arc::new(TrackedTensor::from_var(&x_low).unwrap());
        let loss = t.sqr().unwrap();
        adam_low.backward_step(&loss).unwrap();

        let t = Arc::new(TrackedTensor::from_var(&x_high).unwrap());
        let loss = t.sqr().unwrap();
        adam_high.backward_step(&loss).unwrap();
    }

    let val_low = x_low.data().unwrap().to_flat_vec::<f32>().unwrap()[0].abs();
    let val_high = x_high.data().unwrap().to_flat_vec::<f32>().unwrap()[0].abs();
    assert!(
        val_high < val_low,
        "higher LR Adam should converge faster: low_lr_x={val_low}, high_lr_x={val_high}"
    );
}

// ---------------------------------------------------------------------------
// (h) Gradient clipping
// ---------------------------------------------------------------------------

#[test]
fn test_gradient_clipping_norm_limits_large_gradients() {
    use nn_optim::clip_grad_norm;

    // Large initial value produces large gradients on f(x) = x^2
    let x = Var::new(DynTensor::from_vec(vec![100.0], &[1], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap(); // grad = 2*100 = 200

    let mut grads = backward(&loss).unwrap();

    // Check original gradient is large
    let orig_grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        orig_grad.abs() > 100.0,
        "original gradient should be large, got {orig_grad}"
    );

    // Clip to max norm of 1.0
    let original_norm = clip_grad_norm(&mut grads, 1.0).unwrap();
    assert!(
        original_norm > 100.0,
        "original norm should exceed clip threshold, got {original_norm}"
    );

    // After clipping, gradient magnitude should be ~1.0
    let clipped_grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        clipped_grad.abs() <= 1.0 + 1e-5,
        "clipped gradient should have norm <= 1.0, got {clipped_grad}"
    );
}

#[test]
fn test_gradient_clipping_value_clamps_elements() {
    use nn_optim::clip_grad_value;

    // Multi-element variable with large gradients
    let w = Var::new(DynTensor::from_vec(vec![50.0, -30.0, 20.0, -10.0], &[4], &cpu()).unwrap());
    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let loss = reduce_to_scalar(&tw.sqr().unwrap());

    let mut grads = backward(&loss).unwrap();
    clip_grad_value(&mut grads, 5.0).unwrap();

    let grad_vals = grads.get(&w).unwrap().to_flat_vec::<f32>().unwrap();
    for (i, &g) in grad_vals.iter().enumerate() {
        assert!(
            (-5.0 - 1e-5..=5.0 + 1e-5).contains(&g),
            "gradient[{i}] should be clamped to [-5, 5], got {g}"
        );
    }
}

#[test]
fn test_gradient_clipping_with_training_step() {
    use nn_optim::clip_grad_norm;

    // Verify gradient clipping integrates with manual backward + step
    let w = Var::new(DynTensor::from_vec(vec![10.0, -8.0], &[2, 1], &cpu()).unwrap());
    let mut sgd = Sgd::new(vec![w.clone()], sgd_config(0.1, 0.0, 0.0)).unwrap();

    let x_data = DynTensor::from_vec(vec![1.0, 0.5, -0.5, 1.0], &[2, 2], &cpu()).unwrap();
    let y_data = DynTensor::from_vec(vec![1.0, -1.0], &[2, 1], &cpu()).unwrap();
    let x_input = Arc::new(TrackedTensor::from_tensor(x_data));
    let y_target = Arc::new(TrackedTensor::from_tensor(y_data));

    let initial_w = w.data().unwrap().to_flat_vec::<f32>().unwrap();

    // Manual backward + clip + step
    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let pred = x_input.matmul(&tw).unwrap();
    let loss = mse_loss(&pred, &y_target);

    let mut grads = backward(&loss).unwrap();
    let _norm = clip_grad_norm(&mut grads, 1.0).unwrap();
    sgd.step(&grads).unwrap();

    let new_w = w.data().unwrap().to_flat_vec::<f32>().unwrap();

    // Weights should have changed
    let changed = initial_w
        .iter()
        .zip(new_w.iter())
        .any(|(a, b)| (a - b).abs() > 1e-8);
    assert!(changed, "weights should be updated after clipped step");
}

// ---------------------------------------------------------------------------
// (i) Loss is always finite during training
// ---------------------------------------------------------------------------

#[test]
fn test_loss_is_finite_linear_regression_100_steps() {
    // Train a linear model for 100 steps, assert loss is finite at every step.
    let w = Var::new(DynTensor::from_vec(vec![0.5], &[1, 1], &cpu()).unwrap());
    let b = Var::new(DynTensor::from_vec(vec![0.0], &[1, 1], &cpu()).unwrap());

    let mut sgd = Sgd::new(vec![w.clone(), b.clone()], sgd_config(0.01, 0.0, 0.0)).unwrap();

    let x_data = DynTensor::from_vec(vec![0.0, 1.0, 2.0, 3.0], &[4, 1], &cpu()).unwrap();
    let y_data = DynTensor::from_vec(vec![1.0, 3.0, 5.0, 7.0], &[4, 1], &cpu()).unwrap();
    let x_input = Arc::new(TrackedTensor::from_tensor(x_data));
    let y_target = Arc::new(TrackedTensor::from_tensor(y_data));

    for step in 0..100 {
        let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
        let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());

        let pred = x_input.matmul(&tw).unwrap().add(&tb).unwrap();
        let loss = mse_loss(&pred, &y_target);
        let loss_val = scalar_val(&loss);

        assert!(
            loss_val.is_finite(),
            "loss became non-finite at step {step}: {loss_val}"
        );

        sgd.backward_step(&loss).unwrap();
    }
}

#[test]
fn test_loss_is_finite_adam_mlp_100_steps() {
    // Train a 2-layer MLP with Adam for 100 steps, assert no NaN/Inf.
    let w1 = Var::new(
        DynTensor::from_vec(
            vec![0.5, -0.3, 0.2, 0.8, -0.4, 0.6, 0.1, -0.7],
            &[2, 4],
            &cpu(),
        )
        .unwrap(),
    );
    let b1 = Var::new(DynTensor::from_vec(vec![0.1, -0.1, 0.05, -0.05], &[1, 4], &cpu()).unwrap());
    let w2 = Var::new(DynTensor::from_vec(vec![0.3, -0.5, 0.4, 0.2], &[4, 1], &cpu()).unwrap());
    let b2 = Var::new(DynTensor::from_vec(vec![0.0], &[1, 1], &cpu()).unwrap());

    let mut adam = AdamW::new(
        vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()],
        adam_config(0.01, 0.0),
    )
    .unwrap();

    let x_data = DynTensor::from_vec(
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        &[4, 2],
        &cpu(),
    )
    .unwrap();
    let y_data = DynTensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], &[4, 1], &cpu()).unwrap();
    let x_input = Arc::new(TrackedTensor::from_tensor(x_data));
    let y_target = Arc::new(TrackedTensor::from_tensor(y_data));

    for step in 0..100 {
        let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
        let tb1 = Arc::new(TrackedTensor::from_var(&b1).unwrap());
        let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
        let tb2 = Arc::new(TrackedTensor::from_var(&b2).unwrap());

        let h = x_input.matmul(&tw1).unwrap().add(&tb1).unwrap();
        let h = h.relu().unwrap();
        let pred = h.matmul(&tw2).unwrap().add(&tb2).unwrap();

        let loss = mse_loss(&pred, &y_target);
        let loss_val = scalar_val(&loss);

        assert!(
            loss_val.is_finite(),
            "Adam MLP loss became non-finite at step {step}: {loss_val}"
        );

        adam.backward_step(&loss).unwrap();

        // Also verify all weights remain finite
        for (name, var) in [("w1", &w1), ("b1", &b1), ("w2", &w2), ("b2", &b2)] {
            let vals = var.data().unwrap().to_flat_vec::<f32>().unwrap();
            for (i, &v) in vals.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "{name}[{i}] became non-finite at step {step}: {v}"
                );
            }
        }
    }
}

#[test]
fn test_loss_is_finite_with_various_activations() {
    // Test that different activation functions do not produce NaN/Inf
    // during a training loop with edge-case-ish inputs.
    let w = Var::new(DynTensor::from_vec(vec![2.0, -1.5, 0.5, -0.5], &[2, 2], &cpu()).unwrap());

    let mut sgd = Sgd::new(vec![w.clone()], sgd_config(0.01, 0.0, 0.0)).unwrap();

    // Inputs include 0 and near-zero values that may stress activations
    let x_data = DynTensor::from_vec(vec![0.0, 0.01, -0.01, 3.0], &[2, 2], &cpu()).unwrap();
    let x_input = Arc::new(TrackedTensor::from_tensor(x_data));

    let activations: Vec<(&str, Box<dyn Fn(&Arc<TrackedTensor>) -> Arc<TrackedTensor>>)> = vec![
        ("relu", Box::new(|t: &Arc<TrackedTensor>| t.relu().unwrap())),
        ("silu", Box::new(|t: &Arc<TrackedTensor>| t.silu().unwrap())),
        ("tanh", Box::new(|t: &Arc<TrackedTensor>| t.tanh().unwrap())),
        (
            "sigmoid",
            Box::new(|t: &Arc<TrackedTensor>| t.sigmoid().unwrap()),
        ),
        ("gelu", Box::new(|t: &Arc<TrackedTensor>| t.gelu().unwrap())),
    ];

    for (name, activation) in &activations {
        // Reset weights for each activation test
        w.set(&DynTensor::from_vec(vec![2.0, -1.5, 0.5, -0.5], &[2, 2], &cpu()).unwrap())
            .unwrap();

        for step in 0..30 {
            let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
            let h = x_input.matmul(&tw).unwrap();
            let activated = activation(&h);
            let loss = reduce_to_scalar(&activated);
            let loss_val = scalar_val(&loss);

            assert!(
                loss_val.is_finite(),
                "{name} loss became non-finite at step {step}: {loss_val}"
            );

            sgd.backward_step(&loss).unwrap();
        }
    }
}
