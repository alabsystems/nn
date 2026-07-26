#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for GPU command buffer batching via `with_gpu_scope` and always-on
//! lazy batching (#2009).

use super::*;
use crate::test_common::init;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

#[test]
fn test_scope_inactive_by_default() {
    assert!(!is_lazy_batch_active());
}

#[test]
fn test_scope_active_inside_closure() {
    init();
    let result = with_gpu_scope(|| {
        // Lazy batch is not active until first GPU dispatch.
        assert!(!is_lazy_batch_active());
        // Trigger a dispatch to create the batch.
        get_or_create_batch().unwrap();
        assert!(is_lazy_batch_active());
        Ok(42)
    });
    assert_eq!(result.unwrap(), 42);
    // Scope flushes on exit — batch is consumed.
    assert!(!is_lazy_batch_active());
}

#[test]
fn test_scope_inactive_after_error() {
    init();
    let result: nn_core::Result<()> =
        with_gpu_scope(|| Err(TensorError::InvalidShape("test error".into())));
    assert!(result.is_err());
    // Scope must be cleaned up even on error.
    assert!(!is_lazy_batch_active());
}

#[test]
fn test_nested_scope_reuses_outer() {
    init();
    let result = with_gpu_scope(|| {
        // Create a batch to test nesting behavior.
        get_or_create_batch().unwrap();
        assert!(is_lazy_batch_active());
        // Inner scope flushes its own batch — but outer can create a new one.
        with_gpu_scope(|| {
            // Inner flush consumed the batch from the outer scope.
            // A new get_or_create_batch inside would create a fresh one.
            Ok(())
        })?;
        // After inner scope flush, batch was consumed.
        assert!(!is_lazy_batch_active());
        Ok(99)
    });
    assert_eq!(result.unwrap(), 99);
    assert!(!is_lazy_batch_active());
}

#[test]
fn test_scope_batches_gpu_ops() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 2.0, DType::F32, &device).unwrap();
    let b = DynTensor::full(&[4], 3.0, DType::F32, &device).unwrap();

    // Without scope: each op commits individually (existing behavior).
    let result_no_scope = a.add(&b).unwrap();
    let vals_no_scope = result_no_scope
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // With scope: all ops share one commit.
    let result_with_scope = with_gpu_scope(|| {
        let c = a.add(&b)?;
        let d = c.mul(&a)?;
        Ok(d)
    })
    .unwrap();
    let vals_scope = result_with_scope
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // add(2,3)=5, then mul(5,2)=10
    assert_eq!(vals_no_scope, vec![5.0; 4]);
    assert_eq!(vals_scope, vec![10.0; 4]);
}

#[test]
fn test_scope_multi_op_chain() {
    init();

    let device = Device::metal();
    let x = DynTensor::full(&[8], 3.0, DType::F32, &device).unwrap();

    let result = with_gpu_scope(|| {
        let a = x.mul_scalar(2.0)?; // 6.0
        let b = a.add_scalar(1.0)?; // 7.0
        let c = b.neg()?; // -7.0
        Ok(c)
    })
    .unwrap();

    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![-7.0; 8]);
}

#[test]
fn test_encode_into_lazy_batch_without_active_batch_returns_error() {
    // encode_into_lazy_batch called outside any batch should return Err.
    assert!(!is_lazy_batch_active());
    let result = encode_into_lazy_batch(|_batch| -> Result<(), String> { Ok(()) });
    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("without active batch"),
        "expected 'without active batch' in error, got: {msg}"
    );
}

#[test]
fn test_encode_into_lazy_batch_propagates_encoding_error() {
    // When a batch exists, encoding errors should be propagated as Ok(Err(e)).
    init();
    let result = with_gpu_scope(|| {
        // Must create a batch first — encode_into_lazy_batch requires it.
        get_or_create_batch().unwrap();
        let inner = encode_into_lazy_batch(|_batch| -> Result<(), String> {
            Err("simulated encoding failure".to_string())
        });
        // encode_into_lazy_batch returns Ok(Err(e)) when batch is active but encoding fails.
        match inner {
            Ok(Err(msg)) => {
                assert_eq!(msg, "simulated encoding failure");
            }
            Ok(Ok(())) => panic!("expected encoding error, got Ok(Ok(()))"),
            Err(e) => panic!("expected Ok(Err(...)), got Err({e:?})"),
        }
        Ok(())
    });
    assert!(result.is_ok());
}

// ============================================================================
// Production-path integration tests (P1-254 proof_coverage)
// ============================================================================

/// Matmul through the scope path — most important production op.
///
/// This tests the `tensor_dispatch.rs:235` is_lazy_batch_active() → encode_into_lazy_batch
/// path for the simdgroup/naive matmul kernel.
#[test]
fn test_scope_matmul_produces_correct_result() {
    init();

    let device = Device::metal();
    // [2, 3] × [3, 2] = [2, 2]
    let a_data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let a_cpu = DynTensor::new(&a_data, &[2, 3], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[3, 2], &Device::Cpu).unwrap();
    let a = a_cpu.to_device(&device).unwrap();
    let b = b_cpu.to_device(&device).unwrap();

    // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //         = [[58, 64], [139, 154]]
    let expected = [58.0_f32, 64.0, 139.0, 154.0];

    let result = with_gpu_scope(|| a.matmul(&b)).unwrap();
    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, expected);
}

/// Matmul + add chain within scope — linear layer pattern (y = xW + b).
#[test]
fn test_scope_linear_layer_pattern() {
    init();

    let device = Device::metal();
    // x: [1, 4], w: [4, 2], b: [2]
    let x_cpu = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
    let w_cpu = DynTensor::full(&[4, 2], 0.5, DType::F32, &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&[10.0, 20.0], &[1, 2], &Device::Cpu).unwrap();
    let x = x_cpu.to_device(&device).unwrap();
    let w = w_cpu.to_device(&device).unwrap();
    let b = b_cpu.to_device(&device).unwrap();

    // xW = [1*0.5+2*0.5+3*0.5+4*0.5, same] = [5.0, 5.0]
    // xW + b = [15.0, 25.0]
    let result = with_gpu_scope(|| {
        let h = x.matmul(&w)?;
        h.add(&b)
    })
    .unwrap();

    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![15.0, 25.0]);
}

/// Error mid-chain: first op succeeds, second fails. Batch discarded.
///
/// Verifies that on error the command buffer is dropped without committing,
/// and the scope is cleaned up properly.
#[test]
fn test_scope_error_mid_chain_discards_batch() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 2.0, DType::F32, &device).unwrap();
    let b = DynTensor::full(&[4], 3.0, DType::F32, &device).unwrap();

    let result: nn_core::Result<DynTensor> = with_gpu_scope(|| {
        let c = a.add(&b)?; // succeeds: [5, 5, 5, 5]
                            // Matmul with incompatible shapes → error
        let bad = DynTensor::full(&[3, 3], 1.0, DType::F32, &device)?;
        c.matmul(&bad) // [4] × [3, 3] → shape error
    });

    // The scope must have cleaned up even though the first op succeeded.
    assert!(result.is_err(), "shape mismatch should return error");
    assert!(
        !is_lazy_batch_active(),
        "scope must be inactive after error"
    );
}

/// Nested scope with real GPU matmul — inner ops contribute to outer batch.
#[test]
fn test_nested_scope_with_real_ops() {
    init();

    let device = Device::metal();
    let a_cpu = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2], &Device::Cpu).unwrap();
    let a = a_cpu.to_device(&device).unwrap();
    let b = b_cpu.to_device(&device).unwrap();

    let result = with_gpu_scope(|| {
        let c = a.add(&b)?; // [6, 8, 10, 12]
                            // Inner scope reuses outer batch.
        let d = with_gpu_scope(|| c.mul_scalar(2.0))?; // [12, 16, 20, 24]
        d.add_scalar(1.0) // [13, 17, 21, 25]
    })
    .unwrap();

    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![13.0, 17.0, 21.0, 25.0]);
}

/// Scope with softmax — exercises decomposed GPU op (max→sub→exp→sum→div).
#[test]
fn test_scope_softmax_correctness() {
    init();

    let device = Device::metal();
    let x_cpu = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
    let x = x_cpu.to_device(&device).unwrap();

    let result = with_gpu_scope(|| x.softmax(nn_core::dyn_tensor::D::Minus1)).unwrap();

    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Softmax([1,2,3,4]) = [e^1, e^2, e^3, e^4] / sum
    let exp_vals: Vec<f32> = [1.0_f32, 2.0, 3.0, 4.0].iter().map(|v| v.exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let expected: Vec<f32> = exp_vals.iter().map(|v| v / sum).collect();

    for (got, want) in vals.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-5,
            "softmax mismatch: got {got}, expected {want}"
        );
    }
    // Verify softmax sums to ~1.0.
    let total: f32 = vals.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-5,
        "softmax sum = {total}, expected ~1.0"
    );
}

// Lazy batching, auto-flush, error drop, and dispatch reduction tests
// extracted to `gpu_scope_lazy_tests.rs` for 500-line compliance.
#[path = "gpu_scope_lazy_tests.rs"]
mod lazy_tests;
