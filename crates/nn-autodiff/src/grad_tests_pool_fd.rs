#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference gradient tests for pooling operations.
//!
//! MaxPool1d, MaxPool2d, AvgPool2d, and AdaptiveAvgPool2d backward rules are
//! tested against central-difference numerical gradients.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::tracked::TrackedTensor;
use crate::var::Var;

use super::test_helpers::{check_fd_grad, sum_f64, sum_sqr_f64};

// ---------------------------------------------------------------------------
// MaxPool2d FD tests
// ---------------------------------------------------------------------------

/// MaxPool2d with 2x2 kernel, stride=2, no padding.
/// Input: [1, 1, 4, 4] with distinct values to avoid argmax ties.
#[test]
fn test_max_pool2d_fd_basic() {
    // 4x4 input with all distinct values
    let x_data: Vec<f32> = (1..=16).map(|v| v as f32 * 0.1).collect();
    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 4, 4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.max_pool2d(2, 2, 0).unwrap();
    assert_eq!(y.tensor().dims(), &[1, 1, 2, 2]);

    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(3)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    let fwd = |data: Vec<f32>| -> f64 {
        let v = Var::new(DynTensor::from_vec(data, &[1, 1, 4, 4], &cpu()).unwrap());
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let o = t.max_pool2d(2, 2, 0).unwrap();
        sum_f64(o.tensor())
    };

    check_fd_grad(&gx, &x_data, 1e-3, fwd);
}

/// MaxPool2d with stride=1 (overlapping windows), padding=1.
/// Input: [1, 1, 3, 3] with distinct values.
#[test]
fn test_max_pool2d_fd_overlap_pad() {
    let x_data: Vec<f32> = (1..=9).map(|v| v as f32 * 0.1).collect();
    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 3, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.max_pool2d(2, 1, 0).unwrap();
    // output: [1, 1, 2, 2]
    assert_eq!(y.tensor().dims(), &[1, 1, 2, 2]);

    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(3)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    let fwd = |data: Vec<f32>| -> f64 {
        let v = Var::new(DynTensor::from_vec(data, &[1, 1, 3, 3], &cpu()).unwrap());
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let o = t.max_pool2d(2, 1, 0).unwrap();
        sum_f64(o.tensor())
    };

    check_fd_grad(&gx, &x_data, 1e-3, fwd);
}

/// MaxPool2d with batch=2, channels=2.
#[test]
fn test_max_pool2d_fd_batched() {
    // 2 batches, 2 channels, 4x4 spatial — 64 elements total, all distinct
    let x_data: Vec<f32> = (1..=64).map(|v| v as f32 * 0.01).collect();
    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 2, 4, 4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.max_pool2d(2, 2, 0).unwrap();
    assert_eq!(y.tensor().dims(), &[2, 2, 2, 2]);

    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(3)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    let fwd = |data: Vec<f32>| -> f64 {
        let v = Var::new(DynTensor::from_vec(data, &[2, 2, 4, 4], &cpu()).unwrap());
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let o = t.max_pool2d(2, 2, 0).unwrap();
        sum_f64(o.tensor())
    };

    check_fd_grad(&gx, &x_data, 1e-3, fwd);
}

// ---------------------------------------------------------------------------
// AvgPool2d FD tests
// ---------------------------------------------------------------------------

/// AvgPool2d with 2x2 kernel, stride=2, no padding.
#[test]
fn test_avg_pool2d_fd_basic() {
    let x_data: Vec<f32> = (1..=16).map(|v| v as f32 * 0.1).collect();
    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 4, 4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.avg_pool2d(2, 2, 0).unwrap();
    assert_eq!(y.tensor().dims(), &[1, 1, 2, 2]);

    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(3)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    let fwd = |data: Vec<f32>| -> f64 {
        let v = Var::new(DynTensor::from_vec(data, &[1, 1, 4, 4], &cpu()).unwrap());
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let o = t.avg_pool2d(2, 2, 0).unwrap();
        sum_f64(o.tensor())
    };

    check_fd_grad(&gx, &x_data, 1e-3, fwd);
}

/// AvgPool2d with stride=1 (overlapping windows).
#[test]
fn test_avg_pool2d_fd_overlap() {
    let x_data: Vec<f32> = (1..=9).map(|v| v as f32 * 0.1).collect();
    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 3, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.avg_pool2d(2, 1, 0).unwrap();
    assert_eq!(y.tensor().dims(), &[1, 1, 2, 2]);

    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(3)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    let fwd = |data: Vec<f32>| -> f64 {
        let v = Var::new(DynTensor::from_vec(data, &[1, 1, 3, 3], &cpu()).unwrap());
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let o = t.avg_pool2d(2, 1, 0).unwrap();
        sum_f64(o.tensor())
    };

    check_fd_grad(&gx, &x_data, 1e-3, fwd);
}

/// AvgPool2d with padding=1 and batch=2, channels=2.
#[test]
fn test_avg_pool2d_fd_padded_batched() {
    let x_data: Vec<f32> = (1..=64).map(|v| v as f32 * 0.01).collect();
    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 2, 4, 4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.avg_pool2d(2, 2, 1).unwrap();
    // With padding=1: padded_h=6, out_h = (6-2)/2+1 = 3
    assert_eq!(y.tensor().dims(), &[2, 2, 3, 3]);

    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(3)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    let fwd = |data: Vec<f32>| -> f64 {
        let v = Var::new(DynTensor::from_vec(data, &[2, 2, 4, 4], &cpu()).unwrap());
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let o = t.avg_pool2d(2, 2, 1).unwrap();
        sum_f64(o.tensor())
    };

    check_fd_grad(&gx, &x_data, 1e-3, fwd);
}

/// AvgPool2d with asymmetric input where output_padding_h != output_padding_w.
///
/// Input [1, 1, 6, 5], kernel=3, stride=2, padding=1:
///   base_h = 6+2 = 8, output_padding_h = (8-3) % 2 = 1
///   base_w = 5+2 = 7, output_padding_w = (7-3) % 2 = 0
///
/// The backward code uses `max(1, 0) = 1` with conv_transpose2d (which only
/// accepts a single output_padding) then trims via narrow(). This is correct
/// because the inflated output_padding only admits extra rows/columns BEYOND
/// the trim boundary — values in the kept region are unaffected (#1635).
#[test]
fn test_avg_pool2d_fd_asymmetric_output_padding() {
    // [1, 1, 6, 5] = 30 elements
    let x_data: Vec<f32> = (1..=30).map(|v| v as f32 * 0.1).collect();
    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 6, 5], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.avg_pool2d(3, 2, 1).unwrap();
    // out_h = (6+2-3)/2+1 = 3, out_w = (5+2-3)/2+1 = 3
    assert_eq!(y.tensor().dims(), &[1, 1, 3, 3]);

    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(3)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    let fwd = |data: Vec<f32>| -> f64 {
        let v = Var::new(DynTensor::from_vec(data, &[1, 1, 6, 5], &cpu()).unwrap());
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let o = t.avg_pool2d(3, 2, 1).unwrap();
        sum_f64(o.tensor())
    };

    check_fd_grad(&gx, &x_data, 1e-3, fwd);
}

// ---------------------------------------------------------------------------
// AdaptiveAvgPool2d FD tests
// ---------------------------------------------------------------------------

/// AdaptiveAvgPool2d: 4x4 → 2x2.
#[test]
fn test_adaptive_avg_pool2d_fd_basic() {
    let x_data: Vec<f32> = (1..=16).map(|v| v as f32 * 0.1).collect();
    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 4, 4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.adaptive_avg_pool2d(2, 2).unwrap();
    assert_eq!(y.tensor().dims(), &[1, 1, 2, 2]);

    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(3)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    let fwd = |data: Vec<f32>| -> f64 {
        let v = Var::new(DynTensor::from_vec(data, &[1, 1, 4, 4], &cpu()).unwrap());
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let o = t.adaptive_avg_pool2d(2, 2).unwrap();
        sum_f64(o.tensor())
    };

    check_fd_grad(&gx, &x_data, 1e-3, fwd);
}

/// AdaptiveAvgPool2d: non-square — 6x4 → 3x2.
#[test]
fn test_adaptive_avg_pool2d_fd_non_square() {
    // [1, 1, 6, 4] = 24 elements
    let x_data: Vec<f32> = (1..=24).map(|v| v as f32 * 0.1).collect();
    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 6, 4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.adaptive_avg_pool2d(3, 2).unwrap();
    assert_eq!(y.tensor().dims(), &[1, 1, 3, 2]);

    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(3)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    let fwd = |data: Vec<f32>| -> f64 {
        let v = Var::new(DynTensor::from_vec(data, &[1, 1, 6, 4], &cpu()).unwrap());
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let o = t.adaptive_avg_pool2d(3, 2).unwrap();
        sum_f64(o.tensor())
    };

    check_fd_grad(&gx, &x_data, 1e-3, fwd);
}

/// AdaptiveAvgPool2d: batch=2, channels=2, 4x4 → 1x1 (global average pooling).
#[test]
fn test_adaptive_avg_pool2d_fd_global() {
    let x_data: Vec<f32> = (1..=64).map(|v| v as f32 * 0.01).collect();
    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 2, 4, 4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.adaptive_avg_pool2d(1, 1).unwrap();
    assert_eq!(y.tensor().dims(), &[2, 2, 1, 1]);

    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(3)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    let fwd = |data: Vec<f32>| -> f64 {
        let v = Var::new(DynTensor::from_vec(data, &[2, 2, 4, 4], &cpu()).unwrap());
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let o = t.adaptive_avg_pool2d(1, 1).unwrap();
        sum_f64(o.tensor())
    };

    check_fd_grad(&gx, &x_data, 1e-3, fwd);
}

// ---------------------------------------------------------------------------
// MaxPool1d FD tests
// ---------------------------------------------------------------------------

/// MaxPool1d with kernel=2, stride=2, no padding.
/// Input: [1, 1, 6] with distinct values to avoid argmax ties.
/// Uses sqr() nonlinear loss per #1538 so gradient magnitudes vary per position.
#[test]
fn test_max_pool1d_fd_basic() {
    let x_data: Vec<f32> = (1..=6).map(|v| v as f32 * 0.1).collect();
    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 6], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.max_pool1d(2, 2, 0).unwrap();
    assert_eq!(y.tensor().dims(), &[1, 1, 3]);

    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    let fwd = |data: Vec<f32>| -> f64 {
        let v = Var::new(DynTensor::from_vec(data, &[1, 1, 6], &cpu()).unwrap());
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let o = t.max_pool1d(2, 2, 0).unwrap();
        sum_sqr_f64(o.tensor())
    };

    check_fd_grad(&gx, &x_data, 1e-3, fwd);
}

/// MaxPool1d with stride=1 (overlapping windows).
/// Input: [1, 1, 5] with distinct values.
/// Uses sqr() nonlinear loss per #1538 so gradient magnitudes vary per position.
#[test]
fn test_max_pool1d_fd_overlap() {
    let x_data: Vec<f32> = (1..=5).map(|v| v as f32 * 0.1).collect();
    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 5], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.max_pool1d(3, 1, 0).unwrap();
    assert_eq!(y.tensor().dims(), &[1, 1, 3]);

    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    let fwd = |data: Vec<f32>| -> f64 {
        let v = Var::new(DynTensor::from_vec(data, &[1, 1, 5], &cpu()).unwrap());
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let o = t.max_pool1d(3, 1, 0).unwrap();
        sum_sqr_f64(o.tensor())
    };

    check_fd_grad(&gx, &x_data, 1e-3, fwd);
}

/// MaxPool1d with batch=2, channels=2.
/// Uses sqr() nonlinear loss per #1538 so gradient magnitudes vary per position.
#[test]
fn test_max_pool1d_fd_batched() {
    let x_data: Vec<f32> = (1..=32).map(|v| v as f32 * 0.01).collect();
    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 2, 8], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.max_pool1d(2, 2, 0).unwrap();
    assert_eq!(y.tensor().dims(), &[2, 2, 4]);

    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    let fwd = |data: Vec<f32>| -> f64 {
        let v = Var::new(DynTensor::from_vec(data, &[2, 2, 8], &cpu()).unwrap());
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let o = t.max_pool1d(2, 2, 0).unwrap();
        sum_sqr_f64(o.tensor())
    };

    check_fd_grad(&gx, &x_data, 1e-3, fwd);
}

/// MaxPool1d with padding=1, kernel=3, stride=2.
/// Exercises the padding logic in `tracked_pool_ops.rs` where some kernel
/// positions fall in the padded region and are skipped during argmax.
/// Input: [1, 1, 6] with distinct values.
#[test]
fn test_max_pool1d_fd_padded() {
    let x_data: Vec<f32> = (1..=6).map(|v| v as f32 * 0.1).collect();
    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 6], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    // padded_len = 6 + 2*1 = 8, out_len = (8-3)/2 + 1 = 3
    let y = tx.max_pool1d(3, 2, 1).unwrap();
    assert_eq!(y.tensor().dims(), &[1, 1, 3]);

    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    let fwd = |data: Vec<f32>| -> f64 {
        let v = Var::new(DynTensor::from_vec(data, &[1, 1, 6], &cpu()).unwrap());
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let o = t.max_pool1d(3, 2, 1).unwrap();
        sum_sqr_f64(o.tensor())
    };

    check_fd_grad(&gx, &x_data, 1e-3, fwd);
}
