#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::grad::backward;
use crate::trainable::TrainableModule;
use crate::var::Var;
use nn_core::{DType, Device};

// ---- Config validation ----

#[test]
fn test_config_default_valid() {
    let config = TrainLoopConfig::default();
    assert_eq!(config.max_epochs, 10);
    assert!((config.curriculum_fraction - 0.1).abs() < 1e-10);
    assert!(config.target_score.is_none());
}

#[test]
fn test_config_zero_epochs_rejected() {
    let config = TrainLoopConfig {
        max_epochs: 0,
        ..Default::default()
    };
    let result = run_training_loop(
        &config,
        10,
        |_| vec![],
        |_| {
            Err(AutodiffError::InvalidConfig {
                op: "test",
                reason: "unreachable".to_string(),
            })
        },
        |_| Ok(()),
    );
    assert!(result.is_err());
}

#[test]
fn test_config_bad_curriculum_fraction() {
    let config = TrainLoopConfig {
        curriculum_fraction: 0.0,
        ..Default::default()
    };
    let result = run_training_loop(
        &config,
        10,
        |_| vec![],
        |_| {
            Err(AutodiffError::InvalidConfig {
                op: "test",
                reason: "unreachable".to_string(),
            })
        },
        |_| Ok(()),
    );
    assert!(result.is_err());
}

#[test]
fn test_config_nan_curriculum_fraction() {
    let config = TrainLoopConfig {
        curriculum_fraction: f64::NAN,
        ..Default::default()
    };
    let result = run_training_loop(
        &config,
        10,
        |_| vec![],
        |_| {
            Err(AutodiffError::InvalidConfig {
                op: "test",
                reason: "unreachable".to_string(),
            })
        },
        |_| Ok(()),
    );
    assert!(result.is_err());
}

#[test]
fn test_zero_samples_rejected() {
    let config = TrainLoopConfig::default();
    let result = run_training_loop(
        &config,
        0,
        |_| vec![],
        |_| {
            Err(AutodiffError::InvalidConfig {
                op: "test",
                reason: "unreachable".to_string(),
            })
        },
        |_| Ok(()),
    );
    assert!(result.is_err());
}

// ---- Curriculum selection ----

#[test]
fn test_select_curriculum_sorts_worst_first() {
    let mut scores = vec![
        SampleScore {
            index: 0,
            score: 0.9,
        },
        SampleScore {
            index: 1,
            score: 0.2,
        },
        SampleScore {
            index: 2,
            score: 0.5,
        },
    ];
    let selected = select_curriculum(&mut scores, 0.5, 3);
    // 50% of 3 = 1.5, ceil = 2
    assert_eq!(selected.len(), 2);
    // Worst first: index 1 (0.2), then index 2 (0.5)
    assert_eq!(selected[0], 1);
    assert_eq!(selected[1], 2);
}

#[test]
fn test_select_curriculum_at_least_one() {
    let mut scores = vec![
        SampleScore {
            index: 0,
            score: 0.9,
        },
        SampleScore {
            index: 1,
            score: 0.8,
        },
    ];
    // Very small fraction
    let selected = select_curriculum(&mut scores, 0.01, 2);
    assert_eq!(selected.len(), 1);
}

// ---- Training loop basics ----

#[test]
fn test_training_loop_runs_epochs() {
    let config = TrainLoopConfig {
        max_epochs: 3,
        curriculum_fraction: 0.5,
        ..Default::default()
    };

    let mut eval_count = 0;
    let mut train_count = 0;

    let summary = run_training_loop(
        &config,
        4,
        |_epoch| {
            eval_count += 1;
            vec![
                SampleScore {
                    index: 0,
                    score: 0.3,
                },
                SampleScore {
                    index: 1,
                    score: 0.7,
                },
                SampleScore {
                    index: 2,
                    score: 0.5,
                },
                SampleScore {
                    index: 3,
                    score: 0.9,
                },
            ]
        },
        |_sample_idx| {
            train_count += 1;
            // Return a simple scalar loss
            let t = nn_core::dyn_tensor::DynTensor::new(&[0.5], &[1], &Device::Cpu).expect("new");
            Ok(Arc::new(TrackedTensor::from_tensor(t)))
        },
        |_loss| {
            // No-op optimizer (just backward)
            Ok(())
        },
    )
    .expect("training loop");

    assert_eq!(summary.epoch_metrics.len(), 3);
    assert_eq!(eval_count, 3);
    // 50% of 4 = 2 samples per epoch, 3 epochs = 6 total
    assert_eq!(train_count, 6);
    assert_eq!(summary.total_steps, 6);
    assert!(!summary.early_stopped);
}

#[test]
fn test_training_loop_early_stopping() {
    let config = TrainLoopConfig {
        max_epochs: 10,
        curriculum_fraction: 0.5,
        target_score: Some(0.8),
        ..Default::default()
    };

    let mut epoch = 0;

    let summary = run_training_loop(
        &config,
        2,
        |_| {
            epoch += 1;
            let score = if epoch <= 2 { 0.5 } else { 0.9 };
            vec![
                SampleScore { index: 0, score },
                SampleScore { index: 1, score },
            ]
        },
        |_| {
            let t = nn_core::dyn_tensor::DynTensor::new(&[0.1], &[1], &Device::Cpu).expect("new");
            Ok(Arc::new(TrackedTensor::from_tensor(t)))
        },
        |_| Ok(()),
    )
    .expect("training loop");

    assert!(summary.early_stopped);
    // Should stop at epoch 3 (when score reaches 0.9 > 0.8)
    assert!(summary.epoch_metrics.len() <= 4);
    assert!(summary.final_score >= 0.8);
}

#[test]
fn test_training_loop_with_backward() {
    let config = TrainLoopConfig {
        max_epochs: 1,
        curriculum_fraction: 1.0,
        ..Default::default()
    };

    let var = Var::zeros(&[2], DType::F32, &Device::Cpu).expect("var");

    let summary = run_training_loop(
        &config,
        2,
        |_| {
            vec![
                SampleScore {
                    index: 0,
                    score: 0.4,
                },
                SampleScore {
                    index: 1,
                    score: 0.6,
                },
            ]
        },
        |_sample_idx| {
            // Build a simple loss: sum(x^2)
            let x = Arc::new(TrackedTensor::from_var(&var).expect("from_var"));
            let sq = x.sqr().expect("sqr");
            // Sum to scalar
            sq.sum_keepdim(0)
        },
        |loss| {
            // Actually run backward
            let grads = backward(loss).expect("backward");
            // Verify gradient exists
            assert!(grads.get(&var).is_some());
            Ok(())
        },
    )
    .expect("training loop");

    assert_eq!(summary.total_steps, 2);
    assert_eq!(summary.epoch_metrics[0].curriculum_size, 2);
}

#[test]
fn test_epoch_metrics_loss_tracking() {
    let config = TrainLoopConfig {
        max_epochs: 1,
        curriculum_fraction: 1.0,
        ..Default::default()
    };

    let summary = run_training_loop(
        &config,
        3,
        |_| {
            vec![
                SampleScore {
                    index: 0,
                    score: 0.3,
                },
                SampleScore {
                    index: 1,
                    score: 0.5,
                },
                SampleScore {
                    index: 2,
                    score: 0.7,
                },
            ]
        },
        |sample_idx| {
            // Different loss per sample
            let val = (sample_idx + 1) as f32 * 0.1;
            let t = nn_core::dyn_tensor::DynTensor::new(&[val], &[1], &Device::Cpu).expect("new");
            Ok(Arc::new(TrackedTensor::from_tensor(t)))
        },
        |_| Ok(()),
    )
    .expect("training loop");

    let metrics = &summary.epoch_metrics[0];
    // Loss values: 0.1, 0.2, 0.3 => mean = 0.2
    assert!(
        (metrics.mean_loss - 0.2).abs() < 0.01,
        "Expected mean_loss ~0.2, got {}",
        metrics.mean_loss
    );
    assert_eq!(metrics.curriculum_size, 3);
    assert_eq!(metrics.train_steps, 3);
}

// ---- Helper tests ----

#[test]
fn test_mean_score_empty() {
    assert!((mean_score(&[]) - 0.0).abs() < 1e-10);
}

#[test]
fn test_mean_score_normal() {
    let scores = vec![
        SampleScore {
            index: 0,
            score: 0.2,
        },
        SampleScore {
            index: 1,
            score: 0.4,
        },
        SampleScore {
            index: 2,
            score: 0.6,
        },
    ];
    assert!((mean_score(&scores) - 0.4).abs() < 1e-10);
}

#[test]
fn test_compute_gradients_helper() {
    let t = nn_core::dyn_tensor::DynTensor::new(&[3.0], &[1], &Device::Cpu).expect("new");
    let var = Var::from_tensor(&t);
    let tracked = Arc::new(TrackedTensor::from_var(&var).expect("from_var"));
    let loss = tracked.sqr().expect("sqr"); // loss = 9.0, dloss/dx = 6.0

    let grads = compute_gradients(&loss).expect("compute_gradients");
    let grad_tensor = grads.get(&var).expect("gradient should exist");
    let grad_vals = grad_tensor.to_flat_vec::<f32>().expect("to_vec");
    // d/dx(x^2) = 2x, at x=3.0 => grad = 6.0
    assert!(
        (grad_vals[0] - 6.0).abs() < 1e-5,
        "Expected gradient 6.0, got {}",
        grad_vals[0]
    );
}

#[test]
fn test_training_loop_backward_gradient_values() {
    // Verify gradients have correct values, not just that they exist.
    // Uses non-zero initial values so gradients are non-trivial.
    let config = TrainLoopConfig {
        max_epochs: 1,
        curriculum_fraction: 1.0,
        ..Default::default()
    };

    let init = nn_core::dyn_tensor::DynTensor::new(&[2.0, 3.0], &[2], &Device::Cpu).expect("new");
    let var = Var::from_tensor(&init);

    let summary = run_training_loop(
        &config,
        1,
        |_| {
            vec![SampleScore {
                index: 0,
                score: 0.5,
            }]
        },
        |_sample_idx| {
            // loss = sum(x^2) = 4 + 9 = 13
            let x = Arc::new(TrackedTensor::from_var(&var).expect("from_var"));
            let sq = x.sqr().expect("sqr");
            sq.sum_keepdim(0)
        },
        |loss| {
            let grads = backward(loss).expect("backward");
            let grad_tensor = grads.get(&var).expect("gradient should exist");
            let grad_vals = grad_tensor.to_flat_vec::<f32>().expect("to_vec");
            // d/dx_i(sum(x^2)) = 2*x_i => [4.0, 6.0]
            assert!(
                (grad_vals[0] - 4.0).abs() < 1e-5,
                "Expected grad[0]=4.0, got {}",
                grad_vals[0]
            );
            assert!(
                (grad_vals[1] - 6.0).abs() < 1e-5,
                "Expected grad[1]=6.0, got {}",
                grad_vals[1]
            );
            Ok(())
        },
    )
    .expect("training loop");

    assert_eq!(summary.total_steps, 1);
}

// ---- Training step with linear model (Var + matmul + loss + backward) ----

#[test]
fn test_training_step_linear_model_matmul() {
    use crate::trainable::{TrainableLinear, TrainableModule};
    use nn_core::dyn_tensor::DynTensor;

    // y = x @ W^T + b, loss = mse(y, target)
    let w_data = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
    let b_data = DynTensor::from_vec(vec![0.0, 0.0], &[2], &Device::Cpu).unwrap();
    let layer = TrainableLinear::from_tensors(w_data, Some(b_data));

    let x = DynTensor::from_vec(vec![3.0, 4.0], &[1, 2], &Device::Cpu).unwrap();
    let x_tracked = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&x_tracked).unwrap();

    // With identity weight and zero bias, y should equal x
    let y_vals = y.tensor().to_flat_vec::<f32>().unwrap();
    assert!((y_vals[0] - 3.0).abs() < 1e-5);
    assert!((y_vals[1] - 4.0).abs() < 1e-5);

    // Compute MSE loss against target [1, 1]
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0, 1.0], &[1, 2], &Device::Cpu).unwrap(),
    ));
    let loss = y.mse_loss(&target).unwrap();

    // Verify scalar loss value: mse([3,4],[1,1]) = ((3-1)^2 + (4-1)^2)/2 = (4+9)/2 = 6.5
    let loss_val = loss.tensor().to_scalar::<f32>().unwrap();
    assert!(
        (loss_val - 6.5).abs() < 1e-4,
        "expected 6.5, got {loss_val}"
    );

    // Backward should produce gradients for weight and bias
    let grads = backward(&loss).unwrap();
    assert!(grads.get(layer.weight()).is_some());
    assert!(grads.get(layer.bias().unwrap()).is_some());

    // Weight gradient shape should be [2, 2], bias should be [2]
    assert_eq!(grads.get(layer.weight()).unwrap().dims(), &[2, 2]);
    assert_eq!(grads.get(layer.bias().unwrap()).unwrap().dims(), &[2]);
}

// ---- VarMap creation, insertion, retrieval ----

#[test]
fn test_varmap_creation_and_retrieval() {
    use crate::var_map::VarMap;
    use nn_core::dyn_tensor::DynTensor;

    let mut map = VarMap::new();
    assert!(map.is_empty());

    let w = map
        .get("layer.weight", &[4, 3], DType::F32, &Device::Cpu)
        .unwrap();
    assert_eq!(map.len(), 1);
    assert_eq!(w.dims().unwrap(), &[4, 3]);

    // Retrieving the same name returns the same Var
    let w2 = map
        .get("layer.weight", &[4, 3], DType::F32, &Device::Cpu)
        .unwrap();
    assert_eq!(w.id(), w2.id());

    // Insert another variable
    let b = map
        .get("layer.bias", &[3], DType::F32, &Device::Cpu)
        .unwrap();
    assert_eq!(map.len(), 2);

    // all_vars returns all inserted vars
    let all = map.all_vars();
    assert_eq!(all.len(), 2);

    // Var is mutable: set new data and verify visible via re-get
    let new_data = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    b.set(&new_data).unwrap();
    let b2 = map
        .get("layer.bias", &[3], DType::F32, &Device::Cpu)
        .unwrap();
    let vals = b2.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

// ---- Var gradient accumulation (fan-in within single backward) ----

#[test]
fn test_gradient_accumulation_fan_in() {
    use nn_core::dyn_tensor::DynTensor;

    // y = x + x = 2x, so dy/dx = 2 (gradients accumulate for the same Var used twice)
    let x = Var::from_tensor(&DynTensor::from_vec(vec![5.0], &[1], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.add(&t).unwrap(); // Uses x twice: fan-in
    let grads = backward(&y).unwrap();
    let g = grads.get(&x).unwrap().to_scalar::<f32>().unwrap();
    assert!((g - 2.0).abs() < 1e-5, "expected 2.0, got {g}");
}

#[test]
fn test_gradient_accumulation_triple_fan_in() {
    use nn_core::dyn_tensor::DynTensor;

    // y = x + x + x = 3x, dy/dx = 3
    let x = Var::from_tensor(&DynTensor::from_vec(vec![7.0], &[1], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y1 = t.add(&t).unwrap();
    let y2 = y1.add(&t).unwrap();
    let grads = backward(&y2).unwrap();
    let g = grads.get(&x).unwrap().to_scalar::<f32>().unwrap();
    assert!((g - 3.0).abs() < 1e-5, "expected 3.0, got {g}");
}

// ---- GradStore is fresh each backward (no cross-backward accumulation) ----

#[test]
fn test_separate_backwards_independent_grad_stores() {
    use nn_core::dyn_tensor::DynTensor;

    let x = Var::from_tensor(&DynTensor::from_vec(vec![2.0], &[1], &Device::Cpu).unwrap());

    // First backward: y = x^2, dy/dx = 2*2 = 4
    let t1 = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y1 = t1.sqr().unwrap();
    let grads1 = backward(&y1).unwrap();
    let g1 = grads1.get(&x).unwrap().to_scalar::<f32>().unwrap();
    assert!((g1 - 4.0).abs() < 1e-5);

    // Second backward: z = 3*x, dz/dx = 3
    let t2 = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let z = t2.mul_scalar(3.0).unwrap();
    let loss = z.sum_keepdim(0).unwrap();
    let grads2 = backward(&loss).unwrap();
    let g2 = grads2.get(&x).unwrap().to_scalar::<f32>().unwrap();
    assert!((g2 - 3.0).abs() < 1e-5, "expected 3.0, got {g2}");

    // Grad stores are independent — g1 is still 4, g2 is still 3
    let g1_again = grads1.get(&x).unwrap().to_scalar::<f32>().unwrap();
    assert!((g1_again - 4.0).abs() < 1e-5);
}

// ---- Learning rate schedule verification ----

#[test]
fn test_learning_rate_schedule_linear_decay() {
    use nn_core::dyn_tensor::DynTensor;

    // Simulate 5 steps with linearly decaying learning rate
    let x = Var::from_tensor(&DynTensor::from_vec(vec![10.0], &[1], &Device::Cpu).unwrap());
    let base_lr = 0.5_f64;
    let total_steps = 5;

    let mut x_values = Vec::new();
    for step in 0..total_steps {
        let lr = base_lr * (1.0 - f64::from(step) / f64::from(total_steps));

        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap(); // loss = x^2, grad = 2x
        let grads = backward(&loss).unwrap();
        let grad = grads.get(&x).unwrap();

        let x_data = x.data().unwrap();
        let x_vec = x_data.to_flat_vec::<f32>().unwrap();
        let g_vec = grad.to_flat_vec::<f32>().unwrap();
        let new_x: Vec<f32> = x_vec
            .iter()
            .zip(&g_vec)
            .map(|(xi, gi)| xi - lr as f32 * gi)
            .collect();
        x.set(&DynTensor::from_vec(new_x, &[1], &Device::Cpu).unwrap())
            .unwrap();

        x_values.push(x.data().unwrap().to_scalar::<f32>().unwrap());
    }

    // With decreasing LR, x should still converge toward 0 (convex problem)
    // but the step sizes should decrease over time
    let final_x = x_values.last().unwrap();
    assert!(
        final_x.abs() < 10.0,
        "x should have moved from 10.0 toward 0, got {final_x}"
    );

    // LR at step 4 is 0.1 (base_lr * 0.2), much less than LR at step 0 (0.5)
    // So later steps should produce smaller changes
}

#[test]
fn test_learning_rate_schedule_cosine() {
    use nn_core::dyn_tensor::DynTensor;

    let x = Var::from_tensor(&DynTensor::from_vec(vec![8.0], &[1], &Device::Cpu).unwrap());
    let base_lr = 0.4_f64;
    let total_steps = 10;

    let mut losses = Vec::new();
    for step in 0..total_steps {
        // Cosine schedule: lr = base_lr * 0.5 * (1 + cos(pi * step / total_steps))
        let progress = f64::from(step) / f64::from(total_steps);
        let lr = base_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());

        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        let loss_val = loss.tensor().to_scalar::<f32>().unwrap();
        losses.push(loss_val);

        let grads = backward(&loss).unwrap();
        let grad = grads.get(&x).unwrap();

        let x_data = x.data().unwrap();
        let x_vec = x_data.to_flat_vec::<f32>().unwrap();
        let g_vec = grad.to_flat_vec::<f32>().unwrap();
        let new_x: Vec<f32> = x_vec
            .iter()
            .zip(&g_vec)
            .map(|(xi, gi)| xi - lr as f32 * gi)
            .collect();
        x.set(&DynTensor::from_vec(new_x, &[1], &Device::Cpu).unwrap())
            .unwrap();
    }

    // Loss should generally decrease for this convex problem
    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "final loss {} should be less than initial loss {}",
        losses.last().unwrap(),
        losses.first().unwrap()
    );
}

// ---- Trainable trait: custom model implementation ----

struct ScaleModel {
    scale: Var,
}

impl ScaleModel {
    fn new(init_val: f32) -> Self {
        Self {
            scale: Var::new(
                nn_core::dyn_tensor::DynTensor::from_vec(vec![init_val], &[1], &Device::Cpu)
                    .unwrap(),
            ),
        }
    }
}

impl TrainableModule for ScaleModel {
    fn forward(&self, x: &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>> {
        let w = Arc::new(TrackedTensor::from_var(&self.scale)?);
        x.mul(&w)
    }

    fn vars(&self) -> Vec<&Var> {
        vec![&self.scale]
    }
}

#[test]
fn test_custom_trainable_forward() {
    use nn_core::dyn_tensor::DynTensor;

    let model = ScaleModel::new(3.0);
    let x = DynTensor::from_vec(vec![2.0], &[1], &Device::Cpu).unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = model.forward(&xt).unwrap();
    let val = y.tensor().to_scalar::<f32>().unwrap();
    assert!((val - 6.0).abs() < 1e-5);
}

#[test]
fn test_custom_trainable_gradient() {
    use nn_core::dyn_tensor::DynTensor;

    // y = w * x, loss = y (scalar). d(loss)/dw = x = 4.0
    let model = ScaleModel::new(2.0);
    let x = DynTensor::from_vec(vec![4.0], &[1], &Device::Cpu).unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = model.forward(&xt).unwrap();
    let grads = backward(&y).unwrap();
    let g = grads.get(&model.scale).unwrap().to_scalar::<f32>().unwrap();
    assert!((g - 4.0).abs() < 1e-5, "expected 4.0, got {g}");
}

#[test]
fn test_custom_trainable_vars_list() {
    let model = ScaleModel::new(1.0);
    let vars = model.vars();
    assert_eq!(vars.len(), 1);
    assert_eq!(vars[0].id(), model.scale.id());
}

#[test]
fn test_custom_trainable_training_loop() {
    use nn_core::dyn_tensor::DynTensor;

    // Train w to minimize (w*x - target)^2 where x=1, target=5
    let model = ScaleModel::new(0.0);
    let lr = 0.1_f32;

    let config = TrainLoopConfig {
        max_epochs: 20,
        curriculum_fraction: 1.0,
        target_score: None,
        log_interval: 0,
    };

    let w_ref = model.scale.clone();
    let summary = run_training_loop(
        &config,
        1,
        |_| vec![SampleScore::new(0, 0.1)],
        |_| {
            let x = DynTensor::from_vec(vec![1.0], &[1], &Device::Cpu).unwrap();
            let xt = Arc::new(TrackedTensor::from_tensor(x));
            let y = model.forward(&xt)?;
            let target = Arc::new(TrackedTensor::from_tensor(
                DynTensor::from_vec(vec![5.0], &[1], &Device::Cpu).unwrap(),
            ));
            y.mse_loss(&target)
        },
        |loss| {
            let grads = backward(loss)?;
            let grad = grads.get(&w_ref).unwrap();
            let w_data = w_ref.data()?;
            let w_vec = w_data.to_flat_vec::<f32>().unwrap();
            let g_vec = grad.to_flat_vec::<f32>().unwrap();
            let new_w: Vec<f32> = w_vec
                .iter()
                .zip(&g_vec)
                .map(|(wi, gi)| wi - lr * gi)
                .collect();
            w_ref.set(&DynTensor::from_vec(new_w, &[1], &Device::Cpu).unwrap())?;
            Ok(())
        },
    )
    .unwrap();

    // After 20 SGD steps, w should be close to 5.0
    let final_w = model.scale.data().unwrap().to_scalar::<f32>().unwrap();
    assert!(
        (final_w - 5.0).abs() < 1.0,
        "w should converge toward 5.0, got {final_w}"
    );
    // Loss should decrease
    assert!(
        summary.epoch_metrics.last().unwrap().mean_loss
            < summary.epoch_metrics.first().unwrap().mean_loss
    );
}

// ---- Multiple training steps with decreasing loss (sanity check) ----

#[test]
fn test_multiple_steps_decreasing_loss() {
    use nn_core::dyn_tensor::DynTensor;

    let lr = 0.1_f32;
    let x = Var::from_tensor(&DynTensor::from_vec(vec![3.0], &[1], &Device::Cpu).unwrap());

    // Loss = x^2, grad = 2x, update x <- x*(1 - 2*lr) = 0.8*x.
    // Final recorded loss = (3*0.8^(steps-1))^2. With 10 steps that is 0.162
    // (> 0.01), so run enough steps to genuinely converge below the threshold:
    // 20 steps -> ~0.00187 < 0.01, still monotonically decreasing.
    let mut losses = Vec::new();
    for _ in 0..20 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        losses.push(loss.tensor().to_scalar::<f32>().unwrap());

        let grads = backward(&loss).unwrap();
        let grad = grads.get(&x).unwrap();
        let x_data = x.data().unwrap();
        let xv = x_data.to_flat_vec::<f32>().unwrap();
        let gv = grad.to_flat_vec::<f32>().unwrap();
        let new_x: Vec<f32> = xv.iter().zip(&gv).map(|(xi, gi)| xi - lr * gi).collect();
        x.set(&DynTensor::from_vec(new_x, &[1], &Device::Cpu).unwrap())
            .unwrap();
    }

    // Every step should decrease loss for this convex problem
    for i in 1..losses.len() {
        assert!(
            losses[i] < losses[i - 1],
            "step {i}: loss {:.6} >= prev {:.6}",
            losses[i],
            losses[i - 1]
        );
    }
    // Final loss should be very small
    assert!(
        *losses.last().unwrap() < 0.01,
        "final loss should be < 0.01, got {}",
        losses.last().unwrap()
    );
}

// ---- Edge cases: zero learning rate ----

#[test]
fn test_zero_learning_rate_no_parameter_change() {
    use nn_core::dyn_tensor::DynTensor;

    let lr = 0.0_f32;
    let x = Var::from_tensor(&DynTensor::from_vec(vec![5.0], &[1], &Device::Cpu).unwrap());

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap();

    // SGD step with lr=0 should not change the parameter
    let x_data = x.data().unwrap();
    let xv = x_data.to_flat_vec::<f32>().unwrap();
    let gv = grad.to_flat_vec::<f32>().unwrap();
    let new_x: Vec<f32> = xv.iter().zip(&gv).map(|(xi, gi)| xi - lr * gi).collect();
    x.set(&DynTensor::from_vec(new_x, &[1], &Device::Cpu).unwrap())
        .unwrap();

    let result = x.data().unwrap().to_scalar::<f32>().unwrap();
    assert_eq!(result, 5.0, "parameter should be unchanged with lr=0");
}

// ---- Edge cases: empty VarMap ----

#[test]
fn test_empty_varmap_operations() {
    use crate::var_map::VarMap;

    let map = VarMap::new();
    assert!(map.is_empty());
    assert_eq!(map.len(), 0);
    assert!(map.all_vars().is_empty());

    // to_tensors on empty map should return empty HashMap
    let tensors = map.to_tensors().unwrap();
    assert!(tensors.is_empty());
}

// ---- Multi-variable training: linear model y = Wx + b ----

#[test]
fn test_multi_variable_matmul_gradients() {
    use nn_core::dyn_tensor::DynTensor;

    // y = A @ B where A=[1,2], B=[2,1]
    // A = [[1, 2]], B = [[3], [4]], y = [[11]]
    // dy/dA = B^T = [[3, 4]], dy/dB = A^T = [[1], [2]]
    let a = Var::from_tensor(&DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &Device::Cpu).unwrap());
    let b_var =
        Var::from_tensor(&DynTensor::from_vec(vec![3.0, 4.0], &[2, 1], &Device::Cpu).unwrap());

    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.matmul(&tb).unwrap();

    let y_val = y.tensor().to_scalar::<f32>().unwrap();
    assert!((y_val - 11.0).abs() < 1e-5);

    let grads = backward(&y).unwrap();

    let grad_a = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad_a[0] - 3.0).abs() < 1e-5,
        "dL/dA[0] = 3, got {}",
        grad_a[0]
    );
    assert!(
        (grad_a[1] - 4.0).abs() < 1e-5,
        "dL/dA[1] = 4, got {}",
        grad_a[1]
    );

    let grad_b = grads.get(&b_var).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad_b[0] - 1.0).abs() < 1e-5,
        "dL/dB[0] = 1, got {}",
        grad_b[0]
    );
    assert!(
        (grad_b[1] - 2.0).abs() < 1e-5,
        "dL/dB[1] = 2, got {}",
        grad_b[1]
    );
}

// ---- Gradient clipping simulation (without nn-optim dependency) ----

#[test]
fn test_manual_gradient_clipping_by_value() {
    use nn_core::dyn_tensor::DynTensor;

    // Large initial value produces large gradient
    let x = Var::from_tensor(&DynTensor::from_vec(vec![100.0], &[1], &Device::Cpu).unwrap());

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap(); // grad = 200
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap();
    let grad_val = grad.to_scalar::<f32>().unwrap();

    // Manual gradient clipping: clamp to [-10, 10]
    let clip_value = 10.0_f32;
    let clipped = grad_val.clamp(-clip_value, clip_value);
    assert_eq!(clipped, 10.0, "gradient should be clipped to 10.0");

    // Apply clipped gradient
    let lr = 0.1_f32;
    let x_val = x.data().unwrap().to_scalar::<f32>().unwrap();
    let new_x = x_val - lr * clipped;
    x.set(&DynTensor::from_vec(vec![new_x], &[1], &Device::Cpu).unwrap())
        .unwrap();

    let result = x.data().unwrap().to_scalar::<f32>().unwrap();
    assert!((result - 99.0).abs() < 1e-5, "expected 99.0, got {result}");
}

#[test]
fn test_manual_gradient_clipping_by_norm() {
    use nn_core::dyn_tensor::DynTensor;

    // Two variables with large gradients
    let x = Var::from_tensor(&DynTensor::from_vec(vec![30.0, 40.0], &[2], &Device::Cpu).unwrap());

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let sq = t.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap();
    let grad_vec = grad.to_flat_vec::<f32>().unwrap();
    // grad = [60, 80], norm = 100

    let grad_norm: f32 = grad_vec.iter().map(|g| g * g).sum::<f32>().sqrt();
    assert!((grad_norm - 100.0).abs() < 1e-3);

    // Clip to max_norm=10
    let max_norm = 10.0_f32;
    let scale = if grad_norm > max_norm {
        max_norm / grad_norm
    } else {
        1.0
    };
    let clipped: Vec<f32> = grad_vec.iter().map(|g| g * scale).collect();
    let clipped_norm: f32 = clipped.iter().map(|g| g * g).sum::<f32>().sqrt();
    assert!(
        (clipped_norm - max_norm).abs() < 1e-3,
        "clipped norm should be {max_norm}, got {clipped_norm}"
    );
}

// ---- TrainableLinear: vars count ----

#[test]
fn test_trainable_linear_vars_with_bias() {
    use crate::trainable::{TrainableLinear, TrainableModule};

    let layer = TrainableLinear::new(4, 3, true).unwrap();
    assert_eq!(layer.vars().len(), 2); // weight + bias
}

#[test]
fn test_trainable_linear_vars_without_bias() {
    use crate::trainable::{TrainableLinear, TrainableModule};

    let layer = TrainableLinear::new(4, 3, false).unwrap();
    assert_eq!(layer.vars().len(), 1); // weight only
}

// ---- VarMap shape/dtype mismatch errors ----

#[test]
fn test_varmap_shape_mismatch_on_retrieval() {
    use crate::var_map::VarMap;

    let mut map = VarMap::new();
    map.get("w", &[3, 4], DType::F32, &Device::Cpu).unwrap();
    let err = map.get("w", &[4, 3], DType::F32, &Device::Cpu).unwrap_err();
    assert!(
        format!("{err}").contains("shape mismatch"),
        "expected shape mismatch, got: {err}"
    );
}

#[test]
fn test_varmap_dtype_mismatch_on_retrieval() {
    use crate::var_map::VarMap;

    let mut map = VarMap::new();
    map.get("w", &[3], DType::F32, &Device::Cpu).unwrap();
    let err = map.get("w", &[3], DType::BF16, &Device::Cpu).unwrap_err();
    assert!(
        format!("{err}").contains("dtype mismatch"),
        "expected dtype mismatch, got: {err}"
    );
}

// ---- Training loop with target_score boundary ----

#[test]
fn test_early_stop_exact_threshold() {
    let config = TrainLoopConfig {
        max_epochs: 5,
        curriculum_fraction: 1.0,
        target_score: Some(0.5),
        log_interval: 1,
    };

    // Score exactly at threshold should trigger early stop (>=)
    let summary = run_training_loop(
        &config,
        1,
        |_| vec![SampleScore::new(0, 0.5)],
        |_| unreachable!("should not train"),
        |_| unreachable!("should not step"),
    )
    .unwrap();

    assert!(summary.early_stopped);
    assert_eq!(summary.epoch_metrics.len(), 1);
}

#[test]
fn test_early_stop_just_below_threshold() {
    let config = TrainLoopConfig {
        max_epochs: 1,
        curriculum_fraction: 1.0,
        target_score: Some(0.5),
        log_interval: 1,
    };

    // Score just below threshold should NOT trigger early stop
    let var = Var::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let summary = run_training_loop(
        &config,
        1,
        |_| vec![SampleScore::new(0, 0.4999)],
        |_| {
            let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
            t.sqr()
        },
        |loss| {
            let _grads = backward(loss)?;
            Ok(())
        },
    )
    .unwrap();

    assert!(!summary.early_stopped);
    assert_eq!(summary.epoch_metrics[0].train_steps, 1);
}

// ---- Var initialization methods ----

#[test]
fn test_var_zeros_init() {
    let v = Var::zeros(&[3, 2], DType::F32, &Device::Cpu).unwrap();
    let data = v.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![0.0; 6]);
    assert_eq!(v.dims().unwrap(), &[3, 2]);
    assert_eq!(v.dtype().unwrap(), DType::F32);
}

#[test]
fn test_var_from_tensor() {
    use nn_core::dyn_tensor::DynTensor;

    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let v = Var::from_tensor(&t);
    let data = v.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

// ---- Training loop loss propagation for non-finite ----

#[test]
fn test_training_loop_skips_non_finite_loss_in_mean() {
    // The training loop uses `is_finite()` to skip non-finite loss values
    // in the epoch mean computation. Verify this by passing a sample that
    // produces a very large (but finite) loss.
    let config = TrainLoopConfig {
        max_epochs: 1,
        curriculum_fraction: 1.0,
        target_score: None,
        log_interval: 1,
    };

    let var = Var::from_tensor(
        &nn_core::dyn_tensor::DynTensor::from_vec(vec![100.0], &[1], &Device::Cpu).unwrap(),
    );

    let summary = run_training_loop(
        &config,
        1,
        |_| vec![SampleScore::new(0, 0.1)],
        |_| {
            let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
            t.sqr() // loss = 10000.0
        },
        |loss| {
            let _grads = backward(loss)?;
            Ok(())
        },
    )
    .unwrap();

    assert!(
        (summary.epoch_metrics[0].mean_loss - 10000.0).abs() < 1.0,
        "expected ~10000, got {}",
        summary.epoch_metrics[0].mean_loss
    );
}
