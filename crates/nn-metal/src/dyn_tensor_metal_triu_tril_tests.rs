// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU triu/tril tests.
//!
//! Extracted from `dyn_tensor_metal_ops_tests.rs` for file-size compliance.
//! Validates GPU mask-based triu/tril via where_cond dispatch.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_gpu_vals, init};

#[test]
fn test_gpu_triu_default() {
    init();
    // triu(0) on GPU: builds U8 mask on CPU, transfers to GPU, uses where_cond.
    let t = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
        &Device::metal(),
    )
    .unwrap();
    let r = t.triu(0).unwrap();
    assert_eq!(r.device(), Device::metal(), "triu must preserve GPU device");
    assert_eq!(r.dims(), &[3, 3]);
    assert_gpu_vals(
        &r,
        &[1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0],
        1e-6,
        "triu(0) 3x3",
    );
}

#[test]
fn test_gpu_tril_default() {
    init();
    let t = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
        &Device::metal(),
    )
    .unwrap();
    let r = t.tril(0).unwrap();
    assert_eq!(r.device(), Device::metal(), "tril must preserve GPU device");
    assert_eq!(r.dims(), &[3, 3]);
    assert_gpu_vals(
        &r,
        &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0],
        1e-6,
        "tril(0) 3x3",
    );
}

#[test]
fn test_gpu_triu_positive_diagonal() {
    init();
    // triu(1): zero out strictly below the superdiagonal.
    let t = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
        &Device::metal(),
    )
    .unwrap();
    let r = t.triu(1).unwrap();
    assert_eq!(r.device(), Device::metal());
    assert_gpu_vals(
        &r,
        &[0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0],
        1e-6,
        "triu(1) 3x3",
    );
}

#[test]
fn test_gpu_tril_complement() {
    init();
    // triu(0) + tril(-1) should equal the original (triu keeps diagonal, tril(-1)
    // keeps strictly below).
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let t = DynTensor::new(&data, &[3, 3], &Device::metal()).unwrap();
    let upper = t.triu(0).unwrap();
    let lower = t.tril(-1).unwrap();
    let sum = upper.broadcast_add(&lower).unwrap();
    assert_eq!(sum.device(), Device::metal());
    assert_gpu_vals(&sum, &data, 1e-6, "triu(0)+tril(-1) complement");
}
