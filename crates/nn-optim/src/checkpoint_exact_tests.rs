#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Checkpoint round-trip exact-value preservation tests.
//!
//! Verifies that save → load → save produces bit-identical tensor values
//! for all optimizer state (moments, velocities, factors).
//!
//! Extracted from `checkpoint_tests.rs` for 500-line compliance.

use std::sync::Arc;

use nn_autodiff::{TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;

use crate::checkpoint::OptimizerCheckpoint;
use crate::{AdaFactor, AdaFactorConfig, AdamConfig, AdamW, Optimizer, Sgd, SgdConfig};

use super::cpu;

/// Helper: assert two snapshots have bit-identical tensor values.
fn assert_snapshots_identical(
    snap1: &crate::checkpoint::OptimizerSnapshot,
    snap2: &crate::checkpoint::OptimizerSnapshot,
) {
    assert_eq!(snap1.tensors.len(), snap2.tensors.len());
    for (key, t1) in &snap1.tensors {
        let t2 = snap2
            .tensors
            .get(key)
            .unwrap_or_else(|| panic!("key {key} missing in round-tripped snapshot"));
        let v1 = t1
            .to_flat_vec::<f32>()
            .expect("snap1 tensor to_flat_vec_f32");
        let v2 = t2
            .to_flat_vec::<f32>()
            .expect("snap2 tensor to_flat_vec_f32");
        assert_eq!(v1.len(), v2.len(), "length mismatch for key {key}");
        for (i, (a, b)) in v1.iter().zip(v2.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "bit mismatch at index {i} for key {key}: {a} vs {b}"
            );
        }
    }
    assert_eq!(snap1.metadata["step"], snap2.metadata["step"]);
}

/// Verify that checkpoint save → load → save produces bit-identical tensor values.
///
/// Previous tests only checked step counts, tensor key names, and loss trajectory.
/// This test verifies the actual moment/velocity tensor VALUES survive round-trip.
#[test]
fn test_adamw_checkpoint_roundtrip_exact_values() {
    let v =
        Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).expect("create tensor"));
    let mut adam = AdamW::new(vec![v.clone()], AdamConfig::default()).expect("create adam");

    // Train 5 steps to accumulate non-trivial moment state.
    for _ in 0..5 {
        let t = Arc::new(TrackedTensor::from_var(&v).expect("track var"));
        let loss = t.sqr().expect("sqr").sum_keepdim(0).expect("sum_keepdim");
        adam.backward_step(&loss).expect("backward_step");
    }

    let snap1 = adam.save_checkpoint().expect("save checkpoint");

    // Load into fresh optimizer.
    let mut adam2 = AdamW::new(vec![v], AdamConfig::default()).expect("create adam2");
    adam2.load_checkpoint(&snap1).expect("load checkpoint");

    // Save again from the loaded optimizer.
    let snap2 = adam2.save_checkpoint().expect("save checkpoint 2");

    assert_snapshots_identical(&snap1, &snap2);
}

/// Verify AdaFactor checkpoint round-trip preserves exact tensor values
/// for matrix parameters (factored row/col moments).
#[test]
fn test_adafactor_checkpoint_roundtrip_exact_values() {
    let v = Var::new(
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu())
            .expect("create matrix tensor"),
    );
    let config = AdaFactorConfig {
        beta1: Some(0.9),
        ..AdaFactorConfig::default()
    };
    let mut opt = AdaFactor::new(vec![v.clone()], config.clone()).expect("create adafactor");

    // Train 5 steps to build non-trivial moment state.
    for _ in 0..5 {
        let t = Arc::new(TrackedTensor::from_var(&v).expect("track var"));
        let sq = t.sqr().expect("sqr");
        let loss = sq
            .sum_keepdim(0)
            .expect("sum dim 0")
            .sum_keepdim(1)
            .expect("sum dim 1");
        opt.backward_step(&loss).expect("backward_step");
    }

    let snap1 = opt.save_checkpoint().expect("save checkpoint");

    // Load into fresh optimizer and re-save.
    let mut opt2 = AdaFactor::new(vec![v], config).expect("create adafactor2");
    opt2.load_checkpoint(&snap1).expect("load checkpoint");
    let snap2 = opt2.save_checkpoint().expect("save checkpoint 2");

    assert_snapshots_identical(&snap1, &snap2);
}

/// SGD checkpoint round-trip preserves exact velocity tensor values.
#[test]
fn test_sgd_checkpoint_roundtrip_exact_values() {
    let v =
        Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).expect("create tensor"));
    let config = SgdConfig {
        lr: 0.01,
        momentum: 0.9,
        ..Default::default()
    };
    let mut sgd = Sgd::new(vec![v.clone()], config.clone()).expect("create sgd");

    // Train 3 steps to build momentum.
    for _ in 0..3 {
        let t = Arc::new(TrackedTensor::from_var(&v).expect("track var"));
        let loss = t.sqr().expect("sqr").sum_keepdim(0).expect("sum_keepdim");
        sgd.backward_step(&loss).expect("backward_step");
    }

    let snap1 = sgd.save_checkpoint().expect("save checkpoint");

    // Load into fresh SGD and re-save.
    let mut sgd2 = Sgd::new(vec![v], config).expect("create sgd2");
    sgd2.load_checkpoint(&snap1).expect("load checkpoint");
    let snap2 = sgd2.save_checkpoint().expect("save checkpoint 2");

    assert_snapshots_identical(&snap1, &snap2);
}
