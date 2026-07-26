// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::tracked::TrackedTensor;
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

fn scalar() -> Arc<TrackedTensor> {
    let t = DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap();
    Arc::new(TrackedTensor::from_tensor(t))
}

#[test]
fn test_op_debug_binary() {
    let a = scalar();
    let b = scalar();
    let op = Op::Add(a, b);
    assert_eq!(format!("{op:?}"), "Add");
}

#[test]
fn test_op_debug_unary() {
    let x = scalar();
    let op = Op::Relu(x);
    assert_eq!(format!("{op:?}"), "Relu");
}

#[test]
fn test_op_debug_reduction() {
    let x = scalar();
    let op = Op::SumKeepDim(x, 0);
    assert_eq!(format!("{op:?}"), "SumKeepDim(dim=0)");
}

#[test]
fn test_op_debug_shape() {
    let x = scalar();
    let op = Op::Reshape(x, vec![2, 3]);
    assert_eq!(format!("{op:?}"), "Reshape([2, 3])");
}

#[test]
fn test_op_debug_scalar_op() {
    let x = scalar();
    let op = Op::MulScalar(x, 2.0);
    assert_eq!(format!("{op:?}"), "MulScalar(2)");
}
