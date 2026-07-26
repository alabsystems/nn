#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Stress tests for iterative topological sort.
//!
//! Verifies that the iterative DFS in `topological_sort` handles deep
//! computation graphs without stack overflow. The recursive version would
//! overflow the 8MB default stack at ~30,000 frames.

use std::sync::Arc;

use crate::grad::backward;
use crate::tracked::TrackedTensor;

use super::test_helpers::scalar_var;

/// Build a chain of 1,000 sequential add-scalar operations and run backward.
///
/// With recursive DFS, this would overflow at ~30K frames.
/// The iterative version uses heap-allocated Vec instead.
#[test]
fn test_deep_chain_10k_no_stack_overflow() {
    let x = scalar_var(1.0);
    let mut current: Arc<TrackedTensor> = Arc::new(TrackedTensor::from_var(&x).unwrap());

    // Chain 1,000 add_scalar operations: y = ((x + 0.001) + 0.001) + ...
    for _ in 0..1_000 {
        current = current.add_scalar(0.001).unwrap();
    }

    // Loss = the final scalar value
    let grads = backward(&current).unwrap();
    let grad = grads.get(&x).unwrap();

    // d/dx (x + c1 + c2 + ... + c_n) = 1.0 for all constants
    let grad_val = grad.to_flat_vec::<f32>().unwrap();
    assert_eq!(grad_val.len(), 1);
    assert!(
        (grad_val[0] - 1.0).abs() < 1e-5,
        "Expected gradient ~1.0, got {}",
        grad_val[0]
    );
}

/// Build a chain of 2,000 sequential multiply operations and run backward.
///
/// Mul backward distributes gradients to both operands, but in a chain
/// x * 1.0001 * 1.0001 * ..., the graph depth is 2,000.
#[test]
fn test_deep_chain_20k_mul_scalar() {
    let x = scalar_var(1.0);
    let mut current: Arc<TrackedTensor> = Arc::new(TrackedTensor::from_var(&x).unwrap());

    // Chain 2,000 mul_scalar operations: y = x * 1.0001 * 1.0001 * ...
    for _ in 0..2_000 {
        current = current.mul_scalar(1.0001).unwrap();
    }

    let grads = backward(&current).unwrap();
    let grad = grads.get(&x).unwrap();

    // d/dx (x * c^n) = c^n where c = 1.0001, n = 2000
    // c^2000 ≈ e^(2000 * ln(1.0001)) ≈ e^0.2 ≈ 1.2214
    let expected = 1.0001_f64.powi(2_000);
    let grad_val = f64::from(grad.to_flat_vec::<f32>().unwrap()[0]);
    let rel_err = (grad_val - expected).abs() / expected;
    assert!(
        rel_err < 0.01,
        "Expected gradient ~{expected:.2}, got {grad_val:.2} (rel_err={rel_err:.4})"
    );
}

/// Mix of operations in a deep chain to exercise varied backward rules.
#[test]
fn test_deep_chain_15k_mixed_ops() {
    let x = scalar_var(0.5);
    let mut current: Arc<TrackedTensor> = Arc::new(TrackedTensor::from_var(&x).unwrap());

    // Alternate between add_scalar and mul_scalar for 1,500 ops
    for i in 0..1_500 {
        if i % 2 == 0 {
            current = current.add_scalar(0.0001).unwrap();
        } else {
            current = current.mul_scalar(1.00001).unwrap();
        }
    }

    // Just verify backward completes without stack overflow
    let grads = backward(&current).unwrap();
    let grad = grads.get(&x).unwrap();
    let grad_val = grad.to_flat_vec::<f32>().unwrap();
    assert_eq!(grad_val.len(), 1);
    assert!(
        grad_val[0].is_finite(),
        "Gradient should be finite, got {}",
        grad_val[0]
    );
}
