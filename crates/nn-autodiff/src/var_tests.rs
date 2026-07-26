#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::DType;

#[test]
fn test_var_new_and_data() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let var = Var::new(t);
    let data = var.data().unwrap();
    assert_eq!(data.dims(), &[3]);
    assert_eq!(data.to_flat_vec::<f32>().unwrap(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_var_zeros() {
    let var = Var::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    assert_eq!(var.dims().unwrap(), &[2, 3]);
    assert_eq!(var.dtype().unwrap(), DType::F32);
    let flat = var.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|&v| v == 0.0));
}

#[test]
fn test_var_from_tensor() {
    let t = DynTensor::ones(&[4], DType::F32, &cpu()).unwrap();
    let var = Var::from_tensor(&t);
    assert_eq!(var.data().unwrap().to_flat_vec::<f32>().unwrap(), &[1.0; 4]);
}

#[test]
fn test_var_set_updates_data() {
    let var = Var::zeros(&[2, 2], DType::F32, &cpu()).unwrap();
    let new_data = DynTensor::ones(&[2, 2], DType::F32, &cpu()).unwrap();
    var.set(&new_data).unwrap();
    let flat = var.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|&v| v == 1.0));
}

#[test]
fn test_var_set_rejects_shape_mismatch() {
    let var = Var::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    let wrong_shape = DynTensor::ones(&[3, 2], DType::F32, &cpu()).unwrap();
    let result = var.set(&wrong_shape);
    assert!(result.is_err());
}

#[test]
fn test_var_id_unique() {
    let v1 = Var::zeros(&[1], DType::F32, &cpu()).unwrap();
    let v2 = Var::zeros(&[1], DType::F32, &cpu()).unwrap();
    assert_ne!(v1.id(), v2.id());
}

#[test]
fn test_var_clone_shares_data() {
    let var = Var::zeros(&[3], DType::F32, &cpu()).unwrap();
    let clone = var.clone();
    assert_eq!(var.id(), clone.id());

    // Setting data through the clone is visible from original.
    let new_data = DynTensor::ones(&[3], DType::F32, &cpu()).unwrap();
    clone.set(&new_data).unwrap();
    assert_eq!(var.data().unwrap().to_flat_vec::<f32>().unwrap(), &[1.0; 3]);
}

#[test]
fn test_var_debug() {
    let var = Var::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    let debug = format!("{var:?}");
    assert!(debug.contains("Var"));
    assert!(debug.contains("[2, 3]"));
}

#[test]
fn test_var_device() {
    let var = Var::zeros(&[1], DType::F32, &cpu()).unwrap();
    assert_eq!(var.device().unwrap(), Device::Cpu);
}

#[test]
fn test_var_set_rejects_dtype_mismatch() {
    let var = Var::zeros(&[2], DType::F32, &cpu()).unwrap();
    let wrong_dtype = DynTensor::zeros(&[2], DType::U32, &cpu()).unwrap();
    let result = var.set(&wrong_dtype);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("mismatch"),
        "error should mention mismatch: {err_msg}"
    );
}
