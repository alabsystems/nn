// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `verify_model!` macro.
//!
//! Validates both the verify-only and verify+certify arms expand correctly
//! and produce passing tests on a trivial Linear+ReLU model.
//!
//! Part of #3051, #3020.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::Device;

// -- Arm 1: verify only -------------------------------------------------------

nn_verify::verify_model! {
    name: linear_relu_verify,
    model: {
        let w = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
        Linear::new(w, None).unwrap()
    },
    forward: |m, x| m.forward(&x)?.relu(),
    input: DynTensor::from_vec(vec![0.5, -0.5], &[1, 2], &Device::Cpu).unwrap(),
    bounds: nn_verify::uniform_bounds(&[1, 2], 1.0).unwrap(),
}

// -- Arm 2: verify + certify --------------------------------------------------

nn_verify::verify_model! {
    name: linear_relu_certify,
    model: {
        let w = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
        Linear::new(w, None).unwrap()
    },
    forward: |m, x| m.forward(&x)?.relu(),
    input: DynTensor::from_vec(vec![0.5, -0.5], &[1, 2], &Device::Cpu).unwrap(),
    bounds: nn_verify::uniform_bounds(&[1, 2], 1.0).unwrap(),
    certify: true,
}
