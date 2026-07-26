#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for LoRA adapter.

use super::*;
use nn_autodiff::TrackedTensor;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::Module;
use nn_core::test_utils::{cpu, make_linear, make_linear_with_bias};
use std::sync::Arc;

// ---- AC1: LoraLinear struct ----

#[test]
fn test_lora_linear_creation() {
    let linear = make_linear(8, 4);
    let lora = LoraLinear::from_linear(&linear, 2, 2.0).unwrap();
    assert_eq!(lora.scaling(), 1.0); // alpha / rank = 2.0 / 2
    assert_eq!(lora.lora_a().dims().unwrap(), vec![2, 4]);
    assert_eq!(lora.lora_b().dims().unwrap(), vec![8, 2]);
}

// ---- AC2: from_linear() ----

#[test]
fn test_from_linear_preserves_frozen_weight() {
    let linear = make_linear(8, 4);
    let lora = LoraLinear::from_linear(&linear, 4, 4.0).unwrap();
    let frozen = lora.frozen_weight();
    let original = linear.weight();
    let diff = frozen.sub(original).unwrap();
    let diff_data = diff.to_flat_vec::<f32>().unwrap();
    let max_diff: f32 = diff_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 1e-7, "frozen weight should match original");
}

#[test]
fn test_from_linear_b_initialized_to_zero() {
    let linear = make_linear(8, 4);
    let lora = LoraLinear::from_linear(&linear, 4, 4.0).unwrap();
    let b_data = lora.lora_b().data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        b_data.iter().all(|&v| v == 0.0),
        "B should be zero-initialized"
    );
}

#[test]
fn test_from_linear_a_not_all_zero() {
    let linear = make_linear(8, 4);
    let lora = LoraLinear::from_linear(&linear, 4, 4.0).unwrap();
    let a_data = lora.lora_a().data().unwrap().to_flat_vec::<f32>().unwrap();
    // Random normal — extremely unlikely to be all zeros
    assert!(
        a_data.iter().any(|&v| v != 0.0),
        "A should have non-zero random values"
    );
}

// ---- AC3: Module forward ----

#[test]
fn test_zero_b_matches_original_linear() {
    // When B is zero, LoRA contribution is zero, so output == original Linear
    let linear = make_linear(8, 4);
    let lora = LoraLinear::from_linear(&linear, 4, 4.0).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();
    let y_linear = linear.forward(&x).unwrap();
    let y_lora = lora.forward(&x).unwrap();

    let diff = y_lora.sub(&y_linear).unwrap();
    let diff_data = diff.to_flat_vec::<f32>().unwrap();
    let max_diff: f32 = diff_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-6,
        "zero B LoRA should match original linear, max_diff={max_diff}"
    );
}

#[test]
fn test_forward_with_bias() {
    let linear = make_linear_with_bias(8, 4);
    let lora = LoraLinear::from_linear(&linear, 4, 4.0).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();
    let y_linear = linear.forward(&x).unwrap();
    let y_lora = lora.forward(&x).unwrap();

    let diff = y_lora.sub(&y_linear).unwrap();
    let diff_data = diff.to_flat_vec::<f32>().unwrap();
    let max_diff: f32 = diff_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-6,
        "zero B LoRA with bias should match original, max_diff={max_diff}"
    );
}

#[test]
fn test_forward_batch() {
    let linear = make_linear(4, 3);
    let lora = LoraLinear::from_linear(&linear, 2, 2.0).unwrap();

    // Batch of 3
    let x = DynTensor::new(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
        &cpu(),
    )
    .unwrap();
    let y = lora.forward(&x).unwrap();
    assert_eq!(y.dims(), &[3, 4]);
}

// ---- AC4: merge() ----

#[test]
fn test_merge_with_zero_b() {
    let linear = make_linear(8, 4);
    let lora = LoraLinear::from_linear(&linear, 4, 4.0).unwrap();

    let merged = lora.merge().unwrap();
    let diff = merged.sub(linear.weight()).unwrap();
    let diff_data = diff.to_flat_vec::<f32>().unwrap();
    let max_diff: f32 = diff_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-7,
        "merge with zero B should return original weight, max_diff={max_diff}"
    );
}

#[test]
fn test_merge_numerical_correctness() {
    // Set known A and B to verify merge formula
    let linear = make_linear(2, 3);
    let lora = LoraLinear::from_linear(&linear, 1, 2.0).unwrap();

    // Manually set A = [[1, 0, 0]] (rank=1, in=3) and B = [[1], [0]] (out=2, rank=1)
    let a_new = DynTensor::new(&[1.0, 0.0, 0.0], &[1, 3], &cpu()).unwrap();
    let b_new = DynTensor::new(&[1.0, 0.0], &[2, 1], &cpu()).unwrap();
    lora.lora_a().set(&a_new).unwrap();
    lora.lora_b().set(&b_new).unwrap();

    // merge: W + scaling * B @ A
    // scaling = 2.0 / 1 = 2.0
    // B @ A = [[1], [0]] @ [[1, 0, 0]] = [[1, 0, 0], [0, 0, 0]]
    // scaling * B @ A = [[2, 0, 0], [0, 0, 0]]
    let merged = lora.merge().unwrap();
    let w_data = linear.weight().to_flat_vec::<f32>().unwrap();
    let m_data = merged.to_flat_vec::<f32>().unwrap();

    // Row 0 should have +2.0 in first element
    assert!((m_data[0] - (w_data[0] + 2.0)).abs() < 1e-6);
    assert!((m_data[1] - w_data[1]).abs() < 1e-6);
    assert!((m_data[2] - w_data[2]).abs() < 1e-6);
    // Row 1 should be unchanged
    assert!((m_data[3] - w_data[3]).abs() < 1e-6);
    assert!((m_data[4] - w_data[4]).abs() < 1e-6);
    assert!((m_data[5] - w_data[5]).abs() < 1e-6);
}

// ---- AC5: trainable_vars() ----

#[test]
fn test_trainable_vars_count() {
    let linear = make_linear(8, 4);
    let lora = LoraLinear::from_linear(&linear, 4, 4.0).unwrap();
    let vars = lora.trainable_vars();
    assert_eq!(vars.len(), 2, "should return exactly [A, B]");
}

#[test]
fn test_trainable_vars_shapes() {
    let linear = make_linear(16, 8);
    let lora = LoraLinear::from_linear(&linear, 4, 4.0).unwrap();
    let vars = lora.trainable_vars();
    assert_eq!(
        vars[0].dims().unwrap(),
        vec![4, 8],
        "A should be [rank, in_features]"
    );
    assert_eq!(
        vars[1].dims().unwrap(),
        vec![16, 4],
        "B should be [out_features, rank]"
    );
}

// ---- AC6: LoraConfig ----

#[test]
fn test_lora_config_defaults() {
    let config = LoraConfig::default();
    assert_eq!(config.rank, 8);
    assert!((config.alpha - 8.0).abs() < f64::EPSILON);
    assert_eq!(config.targets, vec!["q_proj", "v_proj"]);
}

#[test]
fn test_lora_config_custom() {
    let config = LoraConfig {
        rank: 16,
        alpha: 32.0,
        targets: vec!["gate_proj".into(), "up_proj".into(), "down_proj".into()],
    };
    assert_eq!(config.rank, 16);
    assert!((config.alpha - 32.0).abs() < f64::EPSILON);
    assert_eq!(config.targets.len(), 3);
}

// ---- Rank-1 produces rank-1 update ----

#[test]
fn test_rank1_lora_produces_rank1_update() {
    let linear = make_linear(4, 3);
    let lora = LoraLinear::from_linear(&linear, 1, 1.0).unwrap();

    // Set known A and B for rank-1
    let a = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let b = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[4, 1], &cpu()).unwrap();
    lora.lora_a().set(&a).unwrap();
    lora.lora_b().set(&b).unwrap();

    let merged = lora.merge().unwrap();
    let delta = merged.sub(linear.weight()).unwrap();
    let delta_data = delta.to_flat_vec::<f32>().unwrap();

    // B @ A = [[1], [2], [3], [4]] @ [[1, 2, 3]] = [[1,2,3], [2,4,6], [3,6,9], [4,8,12]]
    // This is rank 1 — each row is a scalar multiple of [1, 2, 3]
    // scaling = 1.0, so delta == B @ A
    assert!((delta_data[0] - 1.0).abs() < 1e-6);
    assert!((delta_data[1] - 2.0).abs() < 1e-6);
    assert!((delta_data[2] - 3.0).abs() < 1e-6);
    assert!((delta_data[3] - 2.0).abs() < 1e-6);
    assert!((delta_data[4] - 4.0).abs() < 1e-6);
    assert!((delta_data[5] - 6.0).abs() < 1e-6);
}

// ---- Forward with non-zero B (numerical correctness) ----

#[test]
fn test_forward_nonzero_b_numerical() {
    // Set known A and B, verify forward output matches manual computation:
    //   y = x @ W^T + scaling * (x @ A^T @ B^T)
    let linear = make_linear(2, 3); // W = [[0.00, 0.01, 0.02], [0.03, 0.04, 0.05]]
    let lora = LoraLinear::from_linear(&linear, 1, 2.0).unwrap();
    // scaling = 2.0 / 1 = 2.0

    let a = DynTensor::new(&[1.0, 0.0, 0.0], &[1, 3], &cpu()).unwrap();
    let b = DynTensor::new(&[0.5, -0.5], &[2, 1], &cpu()).unwrap();
    lora.lora_a().set(&a).unwrap();
    lora.lora_b().set(&b).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();

    // Base: x @ W^T
    //   W^T = [[0.00, 0.03], [0.01, 0.04], [0.02, 0.05]]
    //   x @ W^T = [1*0.00+2*0.01+3*0.02, 1*0.03+2*0.04+3*0.05]
    //           = [0.08, 0.26]
    //
    // LoRA: scaling * (x @ A^T) @ B^T
    //   A^T = [[1], [0], [0]]
    //   x @ A^T = [1] (shape [1,1])
    //   B^T = [[0.5, -0.5]]
    //   (x @ A^T) @ B^T = [0.5, -0.5] (shape [1,2])
    //   scaled = 2.0 * [0.5, -0.5] = [1.0, -1.0]
    //
    // y = [0.08 + 1.0, 0.26 + (-1.0)] = [1.08, -0.74]

    let y = lora.forward(&x).unwrap();
    let y_data = y.to_flat_vec::<f32>().unwrap();

    assert!(
        (y_data[0] - 1.08).abs() < 1e-5,
        "y[0]: expected 1.08, got {}",
        y_data[0]
    );
    assert!(
        (y_data[1] - (-0.74)).abs() < 1e-5,
        "y[1]: expected -0.74, got {}",
        y_data[1]
    );
}

// ---- Merge-forward equivalence ----

#[test]
fn test_merge_forward_equivalence() {
    // Key LoRA property: merged_linear.forward(x) == lora.forward(x)
    // This ensures merge() and forward() use consistent formulas.
    let linear = make_linear(4, 3);
    let lora = LoraLinear::from_linear(&linear, 2, 4.0).unwrap();

    // Set known A and B
    let a = DynTensor::new(&[1.0, 0.0, -1.0, 0.0, 1.0, 0.0], &[2, 3], &cpu()).unwrap();
    let b = DynTensor::new(&[1.0, 0.0, 0.5, -0.5, -1.0, 1.0, 0.0, 0.5], &[4, 2], &cpu()).unwrap();
    lora.lora_a().set(&a).unwrap();
    lora.lora_b().set(&b).unwrap();

    // Forward through LoRA adapter
    let x = DynTensor::new(&[2.0, -1.0, 3.0], &[1, 3], &cpu()).unwrap();
    let y_lora = lora.forward(&x).unwrap();

    // Forward through merged Linear
    let merged_w = lora.merge().unwrap();
    let merged_linear = Linear::new(merged_w, None).unwrap();
    let y_merged = merged_linear.forward(&x).unwrap();

    // They must match
    let diff = y_lora.sub(&y_merged).unwrap();
    let diff_data = diff.to_flat_vec::<f32>().unwrap();
    let max_diff: f32 = diff_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "merge-forward equivalence failed, max_diff={max_diff}, \
         lora={:?}, merged={:?}",
        y_lora.to_flat_vec::<f32>().unwrap(),
        y_merged.to_flat_vec::<f32>().unwrap()
    );
}

// ---- Bias preservation through from_lora_linear() (regression test for #1500) ----

#[test]
fn test_frozen_bias_accessor() {
    let linear_no_bias = make_linear(4, 3);
    let lora_no_bias = LoraLinear::from_linear(&linear_no_bias, 2, 2.0).unwrap();
    assert!(lora_no_bias.frozen_bias().is_none());

    let linear_with_bias = make_linear_with_bias(4, 3);
    let lora_with_bias = LoraLinear::from_linear(&linear_with_bias, 2, 2.0).unwrap();
    assert!(lora_with_bias.frozen_bias().is_some());
    let bias = lora_with_bias.frozen_bias().unwrap();
    assert_eq!(bias.dims(), &[4]);
}

#[test]
fn test_from_lora_linear_preserves_bias() {
    // Regression test: from_lora_linear() previously dropped the bias (hardcoded None).
    let linear = make_linear_with_bias(4, 8);
    let inference_lora = LoraLinear::from_linear(&linear, 4, 4.0).unwrap();
    assert!(
        inference_lora.frozen_bias().is_some(),
        "inference LoRA should have bias"
    );

    let trainable_lora = TrainableLoraLinear::from_lora_linear(&inference_lora).unwrap();

    // Forward both with same input — outputs must match (bias included in both)
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 8], &cpu()).unwrap();
    let y_inference = inference_lora.forward(&x).unwrap();

    let x_tracked = Arc::new(TrackedTensor::from_tensor(x));
    let y_trainable = trainable_lora.forward(&x_tracked).unwrap();

    let inf_data = y_inference.to_flat_vec::<f32>().unwrap();
    let train_data = y_trainable.tensor().to_flat_vec::<f32>().unwrap();
    for (i, (a, b)) in inf_data.iter().zip(train_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "output mismatch at [{i}]: inference={a}, trainable={b} — bias may be dropped"
        );
    }
}

// Validation regression tests (#1503) extracted to lora_validation_tests.rs
