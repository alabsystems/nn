#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::grad::backward;
use crate::train_loop::{SampleScore, TrainLoopConfig};
use crate::trainable::TrainableLinear;
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

fn make_input(data: &[f32], dims: &[usize]) -> Arc<TrackedTensor> {
    let t = DynTensor::new(data, dims, &Device::Cpu).expect("new");
    Arc::new(TrackedTensor::from_tensor(t))
}

// ---- collect_vars ----

#[test]
fn test_collect_vars_linear_with_bias() {
    let layer = TrainableLinear::new(4, 3, true).expect("linear");
    let vars = collect_vars(&layer);
    assert_eq!(vars.len(), 2); // weight + bias
}

#[test]
fn test_collect_vars_linear_no_bias() {
    let layer = TrainableLinear::new(4, 3, false).expect("linear");
    let vars = collect_vars(&layer);
    assert_eq!(vars.len(), 1); // weight only
}

// ---- count_parameters ----

#[test]
fn test_count_parameters_linear() {
    let layer = TrainableLinear::new(4, 3, true).expect("linear");
    let count = count_parameters(&layer).expect("count");
    // weight: 3*4 = 12, bias: 3
    assert_eq!(count, 15);
}

#[test]
fn test_count_parameters_no_bias() {
    let layer = TrainableLinear::new(4, 3, false).expect("linear");
    let count = count_parameters(&layer).expect("count");
    assert_eq!(count, 12); // weight only: 3*4
}

// ---- verify_forward_finite ----

#[test]
fn test_verify_forward_finite_ok() {
    let layer = TrainableLinear::new(4, 3, false).expect("linear");
    let input = make_input(&[1.0; 8], &[2, 4]);
    verify_forward_finite(&layer, &input).expect("should be finite");
}

// ---- train_with_module ----

#[test]
fn test_train_with_module_basic() {
    let config = TrainLoopConfig {
        max_epochs: 2,
        curriculum_fraction: 1.0,
        ..Default::default()
    };

    let layer = TrainableLinear::new(4, 2, false).expect("linear");

    let summary = train_with_module(
        &config,
        &layer,
        3,
        |_epoch| {
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
        |_sample_idx| {
            let input = make_input(&[1.0; 4], &[1, 4]);
            let reference = make_input(&[0.5, 0.5], &[1, 2]);
            (input, reference)
        },
        |output, reference| {
            // MSE loss
            output.mse_loss(reference)
        },
        |loss| {
            // Just run backward, no optimizer step
            let _grads = backward(loss)?;
            Ok(())
        },
    )
    .expect("training");

    assert_eq!(summary.epoch_metrics.len(), 2);
    assert_eq!(summary.total_steps, 6); // 3 samples * 2 epochs
    assert!(!summary.early_stopped);
}

#[test]
fn test_train_with_module_with_backward() {
    let config = TrainLoopConfig {
        max_epochs: 1,
        curriculum_fraction: 1.0,
        ..Default::default()
    };

    let layer = TrainableLinear::new(4, 2, false).expect("linear");

    let summary = train_with_module(
        &config,
        &layer,
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
            let input = make_input(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
            let reference = make_input(&[1.0, 1.0], &[1, 2]);
            (input, reference)
        },
        TrackedTensor::mse_loss,
        |loss| {
            let grads = backward(loss)?;
            // Verify gradients were computed with non-zero values
            let (_, grad_tensor) = grads
                .var_grads()
                .next()
                .expect("Should have at least one gradient after backward");
            let grad_vals = grad_tensor.to_flat_vec::<f32>().unwrap();
            let grad_norm: f32 = grad_vals.iter().map(|g| g * g).sum::<f32>().sqrt();
            assert!(
                grad_norm > 1e-8,
                "Gradient should be non-zero for non-trivial input/reference, got L2 norm {grad_norm}"
            );
            assert!(
                grad_vals.iter().all(|g| g.is_finite()),
                "All gradient values should be finite"
            );
            Ok(())
        },
    )
    .expect("training");

    assert_eq!(summary.total_steps, 2);
}

#[test]
fn test_train_with_module_early_stopping() {
    let config = TrainLoopConfig {
        max_epochs: 10,
        curriculum_fraction: 0.5,
        target_score: Some(0.8),
        ..Default::default()
    };

    let layer = TrainableLinear::new(2, 1, false).expect("linear");
    let mut epoch = 0;

    let summary = train_with_module(
        &config,
        &layer,
        2,
        |_| {
            epoch += 1;
            let score = if epoch <= 2 { 0.5 } else { 0.9 };
            vec![
                SampleScore { index: 0, score },
                SampleScore { index: 1, score },
            ]
        },
        |_sample_idx| {
            let input = make_input(&[1.0, 2.0], &[1, 2]);
            let reference = make_input(&[0.5], &[1, 1]);
            (input, reference)
        },
        TrackedTensor::mse_loss,
        |loss| {
            let _grads = backward(loss)?;
            Ok(())
        },
    )
    .expect("training");

    assert!(summary.early_stopped);
    assert!(summary.final_score >= 0.8);
}
