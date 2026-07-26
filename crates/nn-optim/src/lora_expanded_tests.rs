// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Expanded LoRA tests covering construction, forward pass, gradient tracking,
//! config injection, low-rank approximation properties, and merge correctness.
//!
//! Complements `lora_tests.rs` (in lora.rs), `lora_trainable_tests.rs`, and
//! `lora_validation_tests.rs` with deeper behavioral and mathematical property tests.

use std::sync::Arc;

use nn_autodiff::{backward, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::test_utils::{cpu, make_linear, make_linear_with_bias};

use crate::adam::{AdamConfig, AdamW};
use crate::lora::{LoraConfig, LoraLinear, TrainableLoraLinear};
use crate::optimizer::Optimizer;

// ============================================================================
// 1. LoraLinear construction with rank
// ============================================================================

#[test]
fn test_lora_linear_construction_various_ranks() {
    let linear = make_linear(16, 8);
    for rank in [1, 2, 4, 8, 16] {
        let lora = LoraLinear::from_linear(&linear, rank, rank as f64).unwrap();
        assert_eq!(lora.lora_a().dims().unwrap(), vec![rank, 8]);
        assert_eq!(lora.lora_b().dims().unwrap(), vec![16, rank]);
        assert!(
            (lora.scaling() - 1.0).abs() < 1e-10,
            "alpha/rank should be 1.0 when alpha == rank"
        );
    }
}

#[test]
fn test_lora_linear_rank_larger_than_dims() {
    // Rank can be larger than both in_features and out_features.
    // This is unusual but valid (over-parameterized LoRA).
    let linear = make_linear(4, 3);
    let lora = LoraLinear::from_linear(&linear, 32, 32.0).unwrap();
    assert_eq!(lora.lora_a().dims().unwrap(), vec![32, 3]);
    assert_eq!(lora.lora_b().dims().unwrap(), vec![4, 32]);
}

// ============================================================================
// 2. LoraLinear forward produces same output as base + delta
// ============================================================================

#[test]
fn test_lora_forward_equals_base_plus_delta() {
    // The LoRA forward y = x @ W^T + scaling * (x @ A^T @ B^T) + bias
    // should match: y = x @ (W + scaling * B @ A)^T + bias
    let linear = make_linear_with_bias(6, 4);
    let lora = LoraLinear::from_linear(&linear, 2, 4.0).unwrap();

    // Set known A and B.
    let a = DynTensor::new(
        &[0.5, -0.3, 0.1, 0.7, 0.2, -0.4, 0.6, -0.1],
        &[2, 4],
        &cpu(),
    )
    .unwrap();
    let b = DynTensor::new(
        &[
            0.3, -0.2, 0.1, 0.5, -0.4, 0.6, 0.7, -0.1, 0.2, -0.3, 0.4, -0.5,
        ],
        &[6, 2],
        &cpu(),
    )
    .unwrap();
    lora.lora_a().set(&a).unwrap();
    lora.lora_b().set(&b).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();

    // Path 1: LoRA forward.
    let y_lora = lora.forward(&x).unwrap();

    // Path 2: Merged weight forward.
    let merged_w = lora.merge().unwrap();
    let merged_linear = Linear::new(merged_w, lora.frozen_bias().cloned()).unwrap();
    let y_merged = merged_linear.forward(&x).unwrap();

    let y_lora_data = y_lora.to_flat_vec::<f32>().unwrap();
    let y_merged_data = y_merged.to_flat_vec::<f32>().unwrap();
    for (i, (l, m)) in y_lora_data.iter().zip(y_merged_data.iter()).enumerate() {
        assert!(
            (l - m).abs() < 1e-5,
            "forward vs merged mismatch at [{i}]: lora={l}, merged={m}"
        );
    }
}

// ============================================================================
// 3. TrainableLoraLinear gradient tracking
// ============================================================================

#[test]
fn test_trainable_lora_gradient_flows_to_a_and_b() {
    let linear = make_linear(4, 6);
    let lora = TrainableLoraLinear::from_linear(&linear, 3, 6.0).unwrap();

    let x_data: Vec<f32> = (0..12).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let x = DynTensor::from_vec(x_data, &[2, 6], &cpu()).unwrap();
    let x_tracked = Arc::new(TrackedTensor::from_tensor(x));

    let y = lora.forward(&x_tracked).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    // Both A and B should have gradients.
    let grad_a = grads.get(lora.lora_a()).expect("A should have gradient");
    let grad_b = grads.get(lora.lora_b()).expect("B should have gradient");
    assert_eq!(grad_a.dims(), &[3, 6], "grad_a shape");
    assert_eq!(grad_b.dims(), &[4, 3], "grad_b shape");

    // Gradients should be finite.
    let a_vals = grad_a.to_flat_vec::<f32>().unwrap();
    assert!(
        a_vals.iter().all(|v| v.is_finite()),
        "grad_a should be finite"
    );
    let b_vals = grad_b.to_flat_vec::<f32>().unwrap();
    assert!(
        b_vals.iter().all(|v| v.is_finite()),
        "grad_b should be finite"
    );
}

#[test]
fn test_trainable_lora_optimizer_updates_a_and_b() {
    let linear = make_linear(3, 4);
    let lora = TrainableLoraLinear::from_linear(&linear, 2, 2.0).unwrap();

    let vars: Vec<Var> = lora.vars().into_iter().cloned().collect();
    let config = AdamConfig {
        lr: 0.01,
        ..AdamConfig::default()
    };
    let mut opt = AdamW::new(vars, config).unwrap();

    let a_before = lora.lora_a().data().unwrap().to_flat_vec::<f32>().unwrap();
    // B starts as zeros.
    let b_before = lora.lora_b().data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        b_before.iter().all(|&v| v == 0.0),
        "B should start as zeros"
    );

    // Run a training step.
    let x = DynTensor::from_vec(vec![1.0f32; 8], &[2, 4], &cpu()).unwrap();
    let x_tracked = Arc::new(TrackedTensor::from_tensor(x));
    let y = lora.forward(&x_tracked).unwrap();
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap();
    opt.backward_step(&loss).unwrap();

    let a_after = lora.lora_a().data().unwrap().to_flat_vec::<f32>().unwrap();
    let b_after = lora.lora_b().data().unwrap().to_flat_vec::<f32>().unwrap();

    // A should have changed (it started non-zero + got gradient).
    let a_changed = a_before
        .iter()
        .zip(a_after.iter())
        .any(|(b, a)| (b - a).abs() > 1e-8);
    assert!(a_changed, "A should change after optimizer step");

    // B should have changed (it started at zero but got gradient).
    let b_changed = b_after.iter().any(|&v| v.abs() > 1e-8);
    assert!(b_changed, "B should change from zero after optimizer step");
}

// ============================================================================
// 4. LoraConfig injection target matching
// ============================================================================

#[test]
fn test_lora_config_default_targets() {
    let config = LoraConfig::default();
    assert_eq!(config.targets, vec!["q_proj", "v_proj"]);
    assert_eq!(config.rank, 8);
    assert!((config.alpha - 8.0).abs() < 1e-10);
}

#[test]
fn test_lora_config_target_matching_simulation() {
    // Simulate how LoraConfig.targets would be used to select layers.
    let config = LoraConfig {
        rank: 4,
        alpha: 8.0,
        targets: vec!["q_proj".into(), "v_proj".into(), "out_proj".into()],
    };

    let layer_names = ["encoder.layer.0.q_proj",
        "encoder.layer.0.k_proj",
        "encoder.layer.0.v_proj",
        "encoder.layer.0.out_proj",
        "encoder.layer.0.mlp.fc1",
        "encoder.layer.0.mlp.fc2"];

    let matched: Vec<_> = layer_names
        .iter()
        .filter(|name| config.targets.iter().any(|t| name.contains(t.as_str())))
        .collect();

    assert_eq!(matched.len(), 3);
    assert!(matched[0].contains("q_proj"));
    assert!(matched[1].contains("v_proj"));
    assert!(matched[2].contains("out_proj"));
}

#[test]
fn test_lora_config_empty_targets() {
    let config = LoraConfig {
        rank: 4,
        alpha: 4.0,
        targets: vec![],
    };
    assert!(config.targets.is_empty());
    // Empty targets means no layers would be adapted.
    let names = ["q_proj", "v_proj"];
    let matched: Vec<_> = names
        .iter()
        .filter(|name| config.targets.iter().any(|t| name.contains(t.as_str())))
        .collect();
    assert!(matched.is_empty());
}

// ============================================================================
// 5. Low rank approximation properties (output is rank-r perturbation)
// ============================================================================

#[test]
fn test_lora_rank_r_perturbation_matrix_rank() {
    // The delta matrix B @ A should have rank at most r.
    // For rank=1: B=[out,1] @ A=[1,in] => outer product => rank 1.
    // Verify by checking all rows are scalar multiples of A.
    let linear = make_linear(5, 4);
    let lora = LoraLinear::from_linear(&linear, 1, 1.0).unwrap();

    let a = DynTensor::new(&[2.0, -1.0, 3.0, 0.5], &[1, 4], &cpu()).unwrap();
    let b = DynTensor::new(&[1.0, -2.0, 0.5, 3.0, -1.5], &[5, 1], &cpu()).unwrap();
    lora.lora_a().set(&a).unwrap();
    lora.lora_b().set(&b).unwrap();

    let merged = lora.merge().unwrap();
    let delta = merged.sub(linear.weight()).unwrap();
    let delta_data = delta.to_flat_vec::<f32>().unwrap();

    // delta = scaling * B @ A. scaling=1.0, so delta = B @ A.
    // Row i of delta should be b[i] * A[0,:].
    let b_vals = [1.0f32, -2.0, 0.5, 3.0, -1.5];
    let a_vals = [2.0f32, -1.0, 3.0, 0.5];

    for (row_idx, &b_val) in b_vals.iter().enumerate() {
        for (col_idx, &a_val) in a_vals.iter().enumerate() {
            let expected = b_val * a_val;
            let actual = delta_data[row_idx * 4 + col_idx];
            assert!(
                (actual - expected).abs() < 1e-5,
                "delta[{row_idx},{col_idx}] = {actual}, expected {expected} (b={b_val}*a={a_val})"
            );
        }
    }
}

#[test]
fn test_lora_rank_2_delta_structure() {
    // Rank-2 LoRA: delta = scaling * B @ A where B=[out,2], A=[2,in].
    // The result is a rank-2 matrix (at most).
    let linear = make_linear(3, 4);
    let lora = LoraLinear::from_linear(&linear, 2, 2.0).unwrap(); // scaling=1.0

    let a = DynTensor::new(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 4], &cpu()).unwrap();
    let b = DynTensor::new(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2], &cpu()).unwrap();
    lora.lora_a().set(&a).unwrap();
    lora.lora_b().set(&b).unwrap();

    let merged = lora.merge().unwrap();
    let delta = merged.sub(linear.weight()).unwrap();
    let delta_data = delta.to_flat_vec::<f32>().unwrap();

    // B @ A = [[1,0],[0,1],[1,1]] @ [[1,0,0,0],[0,1,0,0]]
    //       = [[1,0,0,0],[0,1,0,0],[1,1,0,0]]
    // scaling = 1.0, so delta = B @ A
    let expected = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
    for (i, (&actual, &exp)) in delta_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - exp).abs() < 1e-5,
            "delta[{i}] = {actual}, expected {exp}"
        );
    }
}

// ============================================================================
// 6. Merge LoRA weights back into base linear
// ============================================================================

#[test]
fn test_lora_merge_with_scaling() {
    // Verify merge applies scaling correctly: W_merged = W + (alpha/rank) * B @ A.
    let linear = make_linear(2, 3);
    let lora = LoraLinear::from_linear(&linear, 1, 4.0).unwrap(); // scaling = 4.0/1 = 4.0

    let a = DynTensor::new(&[1.0, 0.0, 0.0], &[1, 3], &cpu()).unwrap();
    let b = DynTensor::new(&[1.0, 0.0], &[2, 1], &cpu()).unwrap();
    lora.lora_a().set(&a).unwrap();
    lora.lora_b().set(&b).unwrap();

    let merged = lora.merge().unwrap();
    let w_data = linear.weight().to_flat_vec::<f32>().unwrap();
    let m_data = merged.to_flat_vec::<f32>().unwrap();

    // B @ A = [[1],[0]] @ [[1,0,0]] = [[1,0,0],[0,0,0]]
    // scaling * B @ A = 4.0 * [[1,0,0],[0,0,0]] = [[4,0,0],[0,0,0]]
    assert!(
        (m_data[0] - (w_data[0] + 4.0)).abs() < 1e-5,
        "merged[0,0] should be W[0,0]+4.0"
    );
    assert!(
        (m_data[1] - w_data[1]).abs() < 1e-5,
        "merged[0,1] should be unchanged"
    );
    assert!(
        (m_data[3] - w_data[3]).abs() < 1e-5,
        "merged[1,0] should be unchanged"
    );
}

#[test]
fn test_trainable_lora_merge_matches_inference_lora_merge() {
    // Both LoraLinear.merge() and TrainableLoraLinear.merge() should produce
    // identical results when sharing the same Var state.
    let linear = make_linear(4, 6);
    let inference_lora = LoraLinear::from_linear(&linear, 3, 6.0).unwrap();

    // Set non-zero A and B.
    let a = DynTensor::new(
        &[
            0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0.7,
            -0.8, 0.9,
        ],
        &[3, 6],
        &cpu(),
    )
    .unwrap();
    let b = DynTensor::new(
        &[
            0.1, 0.2, 0.3, -0.1, -0.2, -0.3, 0.4, 0.5, 0.6, -0.4, -0.5, -0.6,
        ],
        &[4, 3],
        &cpu(),
    )
    .unwrap();
    inference_lora.lora_a().set(&a).unwrap();
    inference_lora.lora_b().set(&b).unwrap();

    let trainable_lora = TrainableLoraLinear::from_lora_linear(&inference_lora).unwrap();

    let merged_inf = inference_lora.merge().unwrap();
    let merged_train = trainable_lora.merge().unwrap();

    let inf_data = merged_inf.to_flat_vec::<f32>().unwrap();
    let train_data = merged_train.to_flat_vec::<f32>().unwrap();
    for (i, (a, b)) in inf_data.iter().zip(train_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "merge mismatch at [{i}]: inference={a}, trainable={b}"
        );
    }
}

// ============================================================================
// 7. LoRA training convergence
// ============================================================================

#[test]
fn test_lora_training_reduces_loss() {
    // Train LoRA on a simple regression task: minimize ||LoRA(x) - target||^2.
    let linear = make_linear(3, 4);
    let lora = TrainableLoraLinear::from_linear(&linear, 2, 4.0).unwrap();

    let vars: Vec<Var> = lora.vars().into_iter().cloned().collect();
    let config = AdamConfig {
        lr: 0.01,
        ..AdamConfig::default()
    };
    let mut opt = AdamW::new(vars, config).unwrap();

    // Fixed input and target.
    let x_data: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let target =
        DynTensor::from_vec(vec![1.0, -1.0, 0.5, 2.0, -0.5, 1.5], &[2, 3], &cpu()).unwrap();

    let mut losses = Vec::new();
    for _ in 0..30 {
        let x = DynTensor::from_vec(x_data.clone(), &[2, 4], &cpu()).unwrap();
        let x_tracked = Arc::new(TrackedTensor::from_tensor(x));
        let target_tracked = Arc::new(TrackedTensor::from_tensor(target.clone()));

        let y = lora.forward(&x_tracked).unwrap();
        let diff = y.sub(&target_tracked).unwrap();
        let loss = diff
            .sqr()
            .unwrap()
            .sum_keepdim(0)
            .unwrap()
            .sum_keepdim(1)
            .unwrap();

        let loss_val = loss
            .tensor()
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum::<f64>();
        losses.push(loss_val);

        opt.backward_step(&loss).unwrap();
    }

    // Loss should decrease over training.
    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "final loss ({}) should be less than initial loss ({})",
        losses.last().unwrap(),
        losses.first().unwrap()
    );

    // Loss should decrease by at least 20%.
    let reduction = 1.0 - losses.last().unwrap() / losses.first().unwrap();
    assert!(
        reduction > 0.2,
        "loss should reduce by at least 20%, got {:.1}% reduction",
        reduction * 100.0
    );
}

// ============================================================================
// 8. LoRA with different scaling values
// ============================================================================

#[test]
fn test_lora_scaling_affects_output_magnitude() {
    // Larger alpha => larger scaling => larger LoRA contribution.
    let linear = make_linear(4, 3);
    let lora_small = LoraLinear::from_linear(&linear, 2, 0.1).unwrap(); // scaling=0.05
    let lora_large = LoraLinear::from_linear(&linear, 2, 100.0).unwrap(); // scaling=50.0

    // Set identical A and B for both.
    let a = DynTensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3], &cpu()).unwrap();
    let b = DynTensor::new(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[4, 2], &cpu()).unwrap();
    lora_small.lora_a().set(&a).unwrap();
    lora_small.lora_b().set(&b).unwrap();
    lora_large.lora_a().set(&a).unwrap();
    lora_large.lora_b().set(&b).unwrap();

    // Compute delta (merged - original) for both.
    let delta_small = lora_small
        .merge()
        .unwrap()
        .sub(linear.weight())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let delta_large = lora_large
        .merge()
        .unwrap()
        .sub(linear.weight())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let norm_small: f64 = delta_small.iter().map(|&v| f64::from(v).powi(2)).sum();
    let norm_large: f64 = delta_large.iter().map(|&v| f64::from(v).powi(2)).sum();

    assert!(
        norm_large > norm_small * 100.0,
        "larger scaling should produce much larger delta: small={norm_small}, large={norm_large}"
    );
}

// ============================================================================
// 9. LoRA alpha=0 disables contribution
// ============================================================================

#[test]
fn test_lora_alpha_zero_disables_contribution() {
    let linear = make_linear(4, 3);
    let lora = LoraLinear::from_linear(&linear, 4, 0.0).unwrap();
    assert_eq!(lora.scaling(), 0.0);

    // Even with non-zero A, the merge should return the original weight.
    // (A is random non-zero from init, B is zero, but scaling=0 means even
    // if B were non-zero, contribution would be zero.)
    let merged = lora.merge().unwrap();
    let diff = merged
        .sub(linear.weight())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let max_diff: f32 = diff.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-7,
        "alpha=0 should disable LoRA contribution, max_diff={max_diff}"
    );
}

// ============================================================================
// 10. Multiple LoRA adapters on same base
// ============================================================================

#[test]
fn test_multiple_lora_adapters_independent() {
    // Two LoRA adapters from the same linear should be independent.
    let linear = make_linear(4, 3);
    let lora1 = LoraLinear::from_linear(&linear, 2, 2.0).unwrap();
    let lora2 = LoraLinear::from_linear(&linear, 2, 2.0).unwrap();

    // Set different B values.
    let b1 = DynTensor::new(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], &[4, 2], &cpu()).unwrap();
    let b2 = DynTensor::new(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], &[4, 2], &cpu()).unwrap();
    lora1.lora_b().set(&b1).unwrap();
    lora2.lora_b().set(&b2).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let y1 = lora1.forward(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let y2 = lora2.forward(&x).unwrap().to_flat_vec::<f32>().unwrap();

    // Outputs should differ (different B values).
    let any_different = y1.iter().zip(y2.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        any_different,
        "different LoRA adapters should produce different outputs"
    );
}

// ============================================================================
// 11. TrainableLoraLinear from_lora_linear shares Vars
// ============================================================================

#[test]
fn test_trainable_from_lora_shares_var_state() {
    let linear = make_linear(4, 3);
    let inference_lora = LoraLinear::from_linear(&linear, 2, 2.0).unwrap();
    let trainable_lora = TrainableLoraLinear::from_lora_linear(&inference_lora).unwrap();

    // Modify B through the trainable adapter.
    let new_b = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2], &cpu()).unwrap();
    trainable_lora.lora_b().set(&new_b).unwrap();

    // The change should be visible through the inference adapter (shared Var).
    let inf_b = inference_lora
        .lora_b()
        .data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let train_b = trainable_lora
        .lora_b()
        .data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(inf_b, train_b, "Var state should be shared");
    assert_eq!(inf_b[0], 1.0);
}

// ============================================================================
// 12. LoRA parameter count is rank * (in + out)
// ============================================================================

#[test]
fn test_lora_parameter_count() {
    let in_features = 768;
    let out_features = 3072;
    let rank = 16;

    let linear = make_linear(out_features, in_features);
    let lora = LoraLinear::from_linear(&linear, rank, rank as f64).unwrap();

    let a_params = lora.lora_a().dims().unwrap().iter().product::<usize>();
    let b_params = lora.lora_b().dims().unwrap().iter().product::<usize>();
    let total_lora_params = a_params + b_params;
    let full_params = in_features * out_features;

    assert_eq!(a_params, rank * in_features);
    assert_eq!(b_params, out_features * rank);
    assert_eq!(total_lora_params, rank * (in_features + out_features));

    // LoRA should use far fewer parameters than full fine-tuning.
    let ratio = total_lora_params as f64 / full_params as f64;
    assert!(
        ratio < 0.03,
        "LoRA with rank={rank} should use <3% of full params, got {:.2}%",
        ratio * 100.0
    );
}
