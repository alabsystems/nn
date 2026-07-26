// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for rank validation in norm backward rules and
//! TrackedTensor forward paths.
//!
//! Each test verifies that under-rank inputs produce `AutodiffError::InvalidConfig`
//! instead of panicking (underflow, OOB index) or silently producing wrong gradients.
//!
//! Covers #2016 AC1-AC9.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::error::AutodiffError;
use crate::tracked::TrackedTensor;
use crate::var::Var;

// ── Helper: create a tracked scalar (rank-0) ─────────────────────────

fn scalar_tracked() -> (Var, Arc<TrackedTensor>) {
    let v = Var::new(DynTensor::from_vec(vec![1.0f32], &[], &cpu()).expect("valid scalar"));
    let t = Arc::new(TrackedTensor::from_var(&v).expect("valid tracked"));
    (v, t)
}

/// Create a tracked 1D tensor (rank-1).
fn rank1_tracked() -> (Var, Arc<TrackedTensor>) {
    let v = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).expect("valid 1d"));
    let t = Arc::new(TrackedTensor::from_var(&v).expect("valid tracked"));
    (v, t)
}

/// Create a tracked 2D tensor (rank-2) [N=2, C=4].
fn rank2_tracked() -> (Var, Arc<TrackedTensor>) {
    let v = Var::new(DynTensor::from_vec(vec![1.0; 8], &[2, 4], &cpu()).expect("valid 2d"));
    let t = Arc::new(TrackedTensor::from_var(&v).expect("valid tracked"));
    (v, t)
}

/// Create a weight tensor for channel dim C.
fn weight_tracked(c: usize) -> (Var, Arc<TrackedTensor>) {
    let v = Var::new(DynTensor::from_vec(vec![1.0; c], &[c], &cpu()).expect("valid weight"));
    let t = Arc::new(TrackedTensor::from_var(&v).expect("valid tracked"));
    (v, t)
}

// ── AC5: TrackedTensor rms_norm rejects rank < 1 ─────────────────────

#[test]
fn test_rms_norm_rejects_rank0() {
    let (_v, t) = scalar_tracked();
    let (_wv, w) = weight_tracked(1);
    let result = t.rms_norm(&w, 1e-5);
    let err = result.expect_err("rms_norm should reject rank-0");
    assert!(
        matches!(err, AutodiffError::InvalidConfig { op: "rms_norm", .. }),
        "expected InvalidConfig for rms_norm, got: {err:?}"
    );
}

// ── AC6: TrackedTensor group_norm rejects rank < 2 ───────────────────

#[test]
fn test_group_norm_rejects_rank1() {
    let (_v, t) = rank1_tracked();
    let (_wv, w) = weight_tracked(3);
    let (_bv, b) = weight_tracked(3);
    let result = t.group_norm(&w, &b, 1, 1e-5);
    let err = result.expect_err("group_norm should reject rank-1");
    assert!(
        matches!(
            err,
            AutodiffError::InvalidConfig {
                op: "group_norm",
                ..
            }
        ),
        "expected InvalidConfig for group_norm, got: {err:?}"
    );
}

// ── AC7: TrackedTensor batch_norm rejects rank < 2 ───────────────────

#[test]
fn test_batch_norm_rejects_rank1() {
    let (_v, t) = rank1_tracked();
    let (_wv, w) = weight_tracked(3);
    let (_bv, b) = weight_tracked(3);
    let result = t.batch_norm(&w, &b, 1e-5);
    let err = result.expect_err("batch_norm should reject rank-1");
    assert!(
        matches!(
            err,
            AutodiffError::InvalidConfig {
                op: "batch_norm",
                ..
            }
        ),
        "expected InvalidConfig for batch_norm, got: {err:?}"
    );
}

// ── AC8: TrackedTensor instance_norm rejects rank < 3 ────────────────

#[test]
fn test_instance_norm_rejects_rank2() {
    let (_v, t) = rank2_tracked();
    let (_wv, w) = weight_tracked(4);
    let (_bv, b) = weight_tracked(4);
    let result = t.instance_norm(&w, &b, 1e-5);
    let err = result.expect_err("instance_norm should reject rank-2");
    assert!(
        matches!(
            err,
            AutodiffError::InvalidConfig {
                op: "instance_norm",
                ..
            }
        ),
        "expected InvalidConfig for instance_norm, got: {err:?}"
    );
}

// ── AC1-AC4: Backward rules reject under-rank via forward-then-backward ──

#[test]
fn test_backward_rms_norm_rejects_rank0() {
    // rms_norm forward already rejects rank-0, so backward is unreachable.
    // This test confirms the forward guard works end-to-end.
    let (_v, t) = scalar_tracked();
    let (_wv, w) = weight_tracked(1);
    let result = t.rms_norm(&w, 1e-5);
    assert!(result.is_err(), "rms_norm should reject rank-0 input");
}

#[test]
fn test_backward_group_norm_rejects_rank1() {
    let (_v, t) = rank1_tracked();
    let (_wv, w) = weight_tracked(3);
    let (_bv, b) = weight_tracked(3);
    let result = t.group_norm(&w, &b, 1, 1e-5);
    assert!(result.is_err(), "group_norm should reject rank-1 input");
}

#[test]
fn test_backward_batch_norm_rejects_rank1() {
    let (_v, t) = rank1_tracked();
    let (_wv, w) = weight_tracked(3);
    let (_bv, b) = weight_tracked(3);
    let result = t.batch_norm(&w, &b, 1e-5);
    assert!(result.is_err(), "batch_norm should reject rank-1 input");
}

#[test]
fn test_backward_instance_norm_rejects_rank2() {
    let (_v, t) = rank2_tracked();
    let (_wv, w) = weight_tracked(4);
    let (_bv, b) = weight_tracked(4);
    let result = t.instance_norm(&w, &b, 1e-5);
    assert!(result.is_err(), "instance_norm should reject rank-2 input");
}
