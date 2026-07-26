#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comparison and selection op GPU tests (ge, gt, lt, le, where_cond).
//!
//! Extracted from `dyn_tensor_metal_nn_tests.rs` (#1299).
//! Verifies GPU comparison ops return GPU tensors and produce correct masks.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

use crate::test_common::init;

/// Convert comparison mask (F32 0.0/1.0 on GPU, U8 on CPU) to `Vec<u8>` for assertion.
fn mask_to_u8_vec(mask: &DynTensor) -> Vec<u8> {
    let cpu_mask = mask.to_device(&Device::Cpu).unwrap();
    if cpu_mask.dtype() == DType::F32 {
        // GPU comparison returns F32 (0.0/1.0) — no round-trip (#1323).
        cpu_mask
            .to_flat_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| v as u8)
            .collect()
    } else {
        let f32_mask = cpu_mask.to_dtype(DType::F32).unwrap();
        f32_mask
            .to_flat_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| v as u8)
            .collect()
    }
}

#[test]
fn test_gpu_ge_returns_gpu_tensor() {
    init();
    let x = DynTensor::new(&[0.5, 1.5, 2.5, 0.0], &[2, 2], &Device::metal()).unwrap();
    let mask = x.ge(1.0).unwrap();
    assert_eq!(
        mask.device(),
        Device::metal(),
        "ge() must return GPU tensor for GPU input"
    );
    // GPU compare returns F32 (0.0/1.0) to avoid round-trip (#1323).
    assert_eq!(mask.dtype(), DType::F32);
    assert_eq!(mask_to_u8_vec(&mask), vec![0, 1, 1, 0]);
}

#[test]
fn test_gpu_gt_returns_gpu_tensor() {
    init();
    let x = DynTensor::new(&[0.5, 1.0, 1.5, 2.0], &[4], &Device::metal()).unwrap();
    let mask = x.gt(1.0).unwrap();
    assert_eq!(
        mask.device(),
        Device::metal(),
        "gt() must return GPU tensor for GPU input"
    );
    assert_eq!(mask_to_u8_vec(&mask), vec![0, 0, 1, 1]);
}

#[test]
fn test_gpu_lt_returns_gpu_tensor() {
    init();
    let x = DynTensor::new(&[0.5, 1.0, 1.5, 2.0], &[4], &Device::metal()).unwrap();
    let mask = x.lt(1.5).unwrap();
    assert_eq!(
        mask.device(),
        Device::metal(),
        "lt() must return GPU tensor for GPU input"
    );
    assert_eq!(mask_to_u8_vec(&mask), vec![1, 1, 0, 0]);
}

#[test]
fn test_gpu_le_returns_gpu_tensor() {
    init();
    let x = DynTensor::new(&[0.5, 1.0, 1.5, 2.0], &[4], &Device::metal()).unwrap();
    let mask = x.le(1.0).unwrap();
    assert_eq!(
        mask.device(),
        Device::metal(),
        "le() must return GPU tensor for GPU input"
    );
    assert_eq!(mask_to_u8_vec(&mask), vec![1, 1, 0, 0]);
}

#[test]
fn test_gpu_where_cond_with_gpu_comparison_mask() {
    init();
    // AC3: where_cond works with GPU input -> comparison mask -> GPU result
    let x = DynTensor::new(&[0.5, 1.5, 2.5, 0.0], &[2, 2], &Device::metal()).unwrap();
    let mask = x.ge(1.0).unwrap();
    assert_eq!(mask.device(), Device::metal());
    let on_true = DynTensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2], &Device::metal()).unwrap();
    let on_false = DynTensor::new(&[-1.0, -2.0, -3.0, -4.0], &[2, 2], &Device::metal()).unwrap();
    let result = mask.where_cond(&on_true, &on_false).unwrap();
    assert_eq!(
        result.device(),
        Device::metal(),
        "where_cond must return GPU tensor"
    );
    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    // ge(1.0): [0.5<1, 1.5>=1, 2.5>=1, 0.0<1] = [0, 1, 1, 0]
    // where: [-1, 20, 30, -4]
    assert_eq!(vals, vec![-1.0, 20.0, 30.0, -4.0]);
}
