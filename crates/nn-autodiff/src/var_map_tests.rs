#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`VarMap`] — extracted from `var_map.rs` for 500-line compliance.

use super::*;
use nn_core::test_utils::cpu;

#[test]
fn test_varmap_get_creates() {
    let mut map = VarMap::new();
    let v = map.get("weight", &[3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(v.data().unwrap().dims(), &[3, 4]);
    assert_eq!(map.len(), 1);
}

#[test]
fn test_varmap_get_same_name_returns_same() {
    let mut map = VarMap::new();
    let v1 = map.get("weight", &[3, 4], DType::F32, &cpu()).unwrap();
    // Same name + same shape/dtype returns the same variable
    let v2 = map.get("weight", &[3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(v1.id(), v2.id());
    assert_eq!(v2.data().unwrap().dims(), &[3, 4]);
}

#[test]
fn test_varmap_get_shape_mismatch_returns_error() {
    let mut map = VarMap::new();
    map.get("weight", &[3, 4], DType::F32, &cpu()).unwrap();
    // Different shape on re-retrieval is an error
    let err = map.get("weight", &[99], DType::F32, &cpu()).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("shape mismatch"),
        "expected shape mismatch error, got: {msg}"
    );
}

#[test]
fn test_varmap_get_dtype_mismatch_returns_error() {
    let mut map = VarMap::new();
    map.get("weight", &[3, 4], DType::F32, &cpu()).unwrap();
    let err = map.get("weight", &[3, 4], DType::BF16, &cpu()).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("dtype mismatch"),
        "expected dtype mismatch error, got: {msg}"
    );
}

#[test]
fn test_varmap_all_vars() {
    let mut map = VarMap::new();
    map.get("a", &[2], DType::F32, &cpu()).unwrap();
    map.get("b", &[3], DType::F32, &cpu()).unwrap();
    map.get("c", &[4], DType::F32, &cpu()).unwrap();
    assert_eq!(map.all_vars().len(), 3);
}

#[test]
fn test_varmap_empty() {
    let map = VarMap::new();
    assert!(map.is_empty());
    assert_eq!(map.len(), 0);
    assert!(map.all_vars().is_empty());
}

#[test]
fn test_varmap_var_is_mutable() {
    let mut map = VarMap::new();
    let v = map.get("w", &[2], DType::F32, &cpu()).unwrap();
    let new_data = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    v.set(&new_data).unwrap();
    // Get again and verify update is visible
    let v2 = map.get("w", &[2], DType::F32, &cpu()).unwrap();
    assert_eq!(
        v2.data().unwrap().to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0]
    );
}

#[test]
fn test_varmap_save_load_roundtrip() {
    let mut map = VarMap::new();
    let w = map.get("weight", &[2, 3], DType::F32, &cpu()).unwrap();
    let b = map.get("bias", &[3], DType::F32, &cpu()).unwrap();

    let w_data = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let b_data = DynTensor::from_vec(vec![0.1, 0.2, 0.3], &[3], &cpu()).unwrap();
    w.set(&w_data).unwrap();
    b.set(&b_data).unwrap();

    let dir = std::env::temp_dir().join(format!("nn_varmap_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    map.save_safetensors(&path).unwrap();

    let mut map2 = VarMap::new();
    map2.get("weight", &[2, 3], DType::F32, &cpu()).unwrap();
    map2.get("bias", &[3], DType::F32, &cpu()).unwrap();
    map2.load_safetensors(&path).unwrap();

    let w2 = map2.to_tensors().unwrap();
    assert_eq!(
        w2["weight"].to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(
        w2["bias"].to_flat_vec::<f32>().unwrap(),
        vec![0.1, 0.2, 0.3]
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_varmap_load_partial() {
    let mut map = VarMap::new();
    let w = map.get("weight", &[2], DType::F32, &cpu()).unwrap();
    w.set(&DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap())
        .unwrap();

    let dir = std::env::temp_dir().join(format!("nn_varmap_partial_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("partial.safetensors");
    map.save_safetensors(&path).unwrap();

    let mut map2 = VarMap::new();
    map2.get("weight", &[2], DType::F32, &cpu()).unwrap();
    map2.get("bias", &[3], DType::F32, &cpu()).unwrap();
    map2.load_safetensors(&path).unwrap();

    let tensors = map2.to_tensors().unwrap();
    assert_eq!(
        tensors["weight"].to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        tensors["bias"].to_flat_vec::<f32>().unwrap(),
        vec![0.0, 0.0, 0.0]
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_varmap_to_tensors() {
    let mut map = VarMap::new();
    map.get("a", &[2], DType::F32, &cpu()).unwrap();
    map.get("b", &[3], DType::F32, &cpu()).unwrap();
    let tensors = map.to_tensors().unwrap();
    assert_eq!(tensors.len(), 2);
    assert!(tensors.contains_key("a"));
    assert!(tensors.contains_key("b"));
}

#[test]
fn test_varmap_load_rejects_nan_tensor() {
    // Save a tensor with NaN, then attempt to load it.
    let dir = std::env::temp_dir().join(format!("nn_varmap_nan_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("nan_weights.safetensors");

    // Write a safetensors file with a NaN tensor
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".to_string(),
        DynTensor::from_vec(vec![1.0, f32::NAN, 3.0], &[3], &cpu()).unwrap(),
    );
    nn_core::dyn_tensor::save_safetensors(&tensors, &path).unwrap();

    // Attempt to load into a VarMap with matching variable
    let mut map = VarMap::new();
    map.get("weight", &[3], DType::F32, &cpu()).unwrap();
    let err = map.load_safetensors(&path).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite") && msg.contains("weight"),
        "expected non-finite error for 'weight', got: {msg}"
    );

    // Verify the variable was NOT updated (still zeros)
    let data = map.to_tensors().unwrap()["weight"]
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(data, vec![0.0, 0.0, 0.0], "variable should remain zeros");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_varmap_load_rejects_inf_tensor() {
    let dir = std::env::temp_dir().join(format!("nn_varmap_inf_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("inf_weights.safetensors");

    let mut tensors = HashMap::new();
    tensors.insert(
        "bias".to_string(),
        DynTensor::from_vec(vec![f32::INFINITY, f32::NEG_INFINITY], &[2], &cpu()).unwrap(),
    );
    nn_core::dyn_tensor::save_safetensors(&tensors, &path).unwrap();

    let mut map = VarMap::new();
    map.get("bias", &[2], DType::F32, &cpu()).unwrap();
    let err = map.load_safetensors(&path).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite") && msg.contains("bias"),
        "expected non-finite error for 'bias', got: {msg}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

// --- New tests for expanded VarMap coverage ---

#[test]
fn test_varmap_default_is_empty() {
    let map = VarMap::default();
    assert!(map.is_empty());
    assert_eq!(map.len(), 0);
}

#[test]
fn test_varmap_get_zero_initialized() {
    // VarMap::get creates zero-initialized variables by default.
    let mut map = VarMap::new();
    let v = map.get("w", &[3, 2], DType::F32, &cpu()).unwrap();
    let data = v.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(data.len(), 6);
    assert!(data.iter().all(|&x| x == 0.0), "expected all zeros");
}

#[test]
fn test_varmap_multiple_distinct_names() {
    let mut map = VarMap::new();
    let v1 = map
        .get("layer1.weight", &[4, 3], DType::F32, &cpu())
        .unwrap();
    let v2 = map.get("layer1.bias", &[4], DType::F32, &cpu()).unwrap();
    let v3 = map
        .get("layer2.weight", &[2, 4], DType::F32, &cpu())
        .unwrap();
    assert_eq!(map.len(), 3);
    // All have distinct IDs
    assert_ne!(v1.id(), v2.id());
    assert_ne!(v2.id(), v3.id());
    assert_ne!(v1.id(), v3.id());
}

#[test]
fn test_varmap_get_bf16_dtype() {
    let mut map = VarMap::new();
    let v = map.get("w", &[2, 2], DType::BF16, &cpu()).unwrap();
    assert_eq!(v.dtype().unwrap(), DType::BF16);
    assert_eq!(v.dims().unwrap(), vec![2, 2]);
}

#[test]
fn test_varmap_get_f16_dtype() {
    let mut map = VarMap::new();
    let v = map.get("w", &[5], DType::F16, &cpu()).unwrap();
    assert_eq!(v.dtype().unwrap(), DType::F16);
}

#[test]
fn test_varmap_get_u32_dtype() {
    let mut map = VarMap::new();
    let v = map.get("indices", &[3], DType::U32, &cpu()).unwrap();
    assert_eq!(v.dtype().unwrap(), DType::U32);
    assert_eq!(v.dims().unwrap(), vec![3]);
}

#[test]
fn test_varmap_all_vars_shapes_match() {
    let mut map = VarMap::new();
    map.get("a", &[2, 3], DType::F32, &cpu()).unwrap();
    map.get("b", &[4], DType::F32, &cpu()).unwrap();
    map.get("c", &[1, 1, 1], DType::F32, &cpu()).unwrap();
    let vars = map.all_vars();
    let mut dims_set: Vec<Vec<usize>> = vars.iter().map(|v| v.dims().unwrap()).collect();
    dims_set.sort();
    assert_eq!(dims_set, vec![vec![1, 1, 1], vec![2, 3], vec![4]]);
}

#[test]
fn test_varmap_empty_string_name() {
    // Empty string is a valid HashMap key.
    let mut map = VarMap::new();
    let v = map.get("", &[2], DType::F32, &cpu()).unwrap();
    assert_eq!(v.dims().unwrap(), vec![2]);
    assert_eq!(map.len(), 1);
    // Retrieving with the same empty name returns the same Var.
    let v2 = map.get("", &[2], DType::F32, &cpu()).unwrap();
    assert_eq!(v.id(), v2.id());
}

#[test]
fn test_varmap_very_long_name() {
    let long_name = "a".repeat(10_000);
    let mut map = VarMap::new();
    let v = map.get(&long_name, &[1], DType::F32, &cpu()).unwrap();
    assert_eq!(v.dims().unwrap(), vec![1]);
    let v2 = map.get(&long_name, &[1], DType::F32, &cpu()).unwrap();
    assert_eq!(v.id(), v2.id());
}

#[test]
fn test_varmap_special_characters_in_name() {
    let mut map = VarMap::new();
    let names = [
        "model/layer.0/weight:0",
        "encoder.layers[3].self_attn.q_proj",
        "weights/block-1/conv",
        "param with spaces",
    ];
    for name in &names {
        map.get(name, &[2], DType::F32, &cpu()).unwrap();
    }
    assert_eq!(map.len(), names.len());
}

#[test]
fn test_varmap_multiple_maps_independent() {
    // Two VarMaps do not share state.
    let mut map1 = VarMap::new();
    let mut map2 = VarMap::new();

    let v1 = map1.get("weight", &[3], DType::F32, &cpu()).unwrap();
    let v2 = map2.get("weight", &[3], DType::F32, &cpu()).unwrap();

    // Different VarMap instances produce different Var IDs
    assert_ne!(v1.id(), v2.id());

    // Mutating through one does not affect the other
    let new_data = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    v1.set(&new_data).unwrap();

    let d1 = map1.to_tensors().unwrap()["weight"]
        .to_flat_vec::<f32>()
        .unwrap();
    let d2 = map2.to_tensors().unwrap()["weight"]
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(d1, vec![1.0, 2.0, 3.0]);
    assert_eq!(d2, vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_varmap_multiple_maps_different_shapes_same_name() {
    // Two maps can have the same name with different shapes.
    let mut map1 = VarMap::new();
    let mut map2 = VarMap::new();

    map1.get("w", &[2, 3], DType::F32, &cpu()).unwrap();
    map2.get("w", &[5, 5], DType::F32, &cpu()).unwrap();

    assert_eq!(map1.to_tensors().unwrap()["w"].dims(), &[2, 3]);
    assert_eq!(map2.to_tensors().unwrap()["w"].dims(), &[5, 5]);
}

#[test]
fn test_varmap_scalar_variable() {
    let mut map = VarMap::new();
    let v = map.get("lr", &[], DType::F32, &cpu()).unwrap();
    assert_eq!(v.dims().unwrap(), Vec::<usize>::new());
    let data = v.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![0.0]);
}

#[test]
fn test_varmap_high_rank_variable() {
    let mut map = VarMap::new();
    let v = map.get("conv", &[2, 3, 4, 5], DType::F32, &cpu()).unwrap();
    assert_eq!(v.dims().unwrap(), vec![2, 3, 4, 5]);
    assert_eq!(v.data().unwrap().to_flat_vec::<f32>().unwrap().len(), 120);
}

#[test]
fn test_varmap_get_idempotent_repeated_calls() {
    // Multiple re-retrieval calls are all idempotent.
    let mut map = VarMap::new();
    let v0 = map.get("x", &[4], DType::F32, &cpu()).unwrap();
    for _ in 0..10 {
        let vi = map.get("x", &[4], DType::F32, &cpu()).unwrap();
        assert_eq!(v0.id(), vi.id());
    }
    assert_eq!(map.len(), 1);
}

#[test]
fn test_varmap_all_vars_for_optimizer() {
    // Simulates optimizer parameter extraction pattern.
    let mut map = VarMap::new();
    map.get("encoder.weight", &[128, 64], DType::F32, &cpu())
        .unwrap();
    map.get("encoder.bias", &[128], DType::F32, &cpu()).unwrap();
    map.get("decoder.weight", &[64, 128], DType::F32, &cpu())
        .unwrap();
    map.get("decoder.bias", &[64], DType::F32, &cpu()).unwrap();

    let all = map.all_vars();
    assert_eq!(all.len(), 4);

    // Each Var data is accessible
    for var in &all {
        let data = var.data().unwrap();
        assert!(!data.dims().is_empty());
    }
}

#[test]
fn test_varmap_gradient_tracking_through_vars() {
    // Vars from VarMap participate in the computation graph.
    use crate::grad::backward;
    use crate::tracked::TrackedTensor;
    use std::sync::Arc;

    let mut map = VarMap::new();
    let w = map.get("w", &[1], DType::F32, &cpu()).unwrap();
    // Set w = [3.0]
    w.set(&DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap())
        .unwrap();

    // Build computation: loss = w^2 => d(loss)/dw = 2*w = 6.0
    let tracked_w = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let loss = tracked_w.sqr().unwrap();

    let grads = backward(&loss).unwrap();
    let grad = grads.get(&w).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 6.0).abs() < 1e-5,
        "expected gradient 6.0, got {}",
        grad[0]
    );
}

#[test]
fn test_varmap_gradient_tracking_two_vars() {
    // Two vars from the same VarMap in a computation graph.
    use crate::grad::backward;
    use crate::tracked::TrackedTensor;
    use std::sync::Arc;

    let mut map = VarMap::new();
    let a = map.get("a", &[1], DType::F32, &cpu()).unwrap();
    let b = map.get("b", &[1], DType::F32, &cpu()).unwrap();
    a.set(&DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap())
        .unwrap();
    b.set(&DynTensor::from_vec(vec![5.0], &[1], &cpu()).unwrap())
        .unwrap();

    // loss = a * b => d(loss)/da = b = 5.0, d(loss)/db = a = 2.0
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let loss = ta.mul(&tb).unwrap();

    let grads = backward(&loss).unwrap();
    let grad_a = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let grad_b = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    assert!(
        (grad_a[0] - 5.0).abs() < 1e-5,
        "expected grad_a=5.0, got {}",
        grad_a[0]
    );
    assert!(
        (grad_b[0] - 2.0).abs() < 1e-5,
        "expected grad_b=2.0, got {}",
        grad_b[0]
    );
}

#[test]
fn test_varmap_to_tensors_empty() {
    let map = VarMap::new();
    let tensors = map.to_tensors().unwrap();
    assert!(tensors.is_empty());
}

#[test]
fn test_varmap_not_empty_after_insert() {
    let mut map = VarMap::new();
    assert!(map.is_empty());
    map.get("x", &[1], DType::F32, &cpu()).unwrap();
    assert!(!map.is_empty());
    assert_eq!(map.len(), 1);
}

#[test]
fn test_varmap_dtype_mismatch_error_contains_name() {
    let mut map = VarMap::new();
    map.get("nn_param", &[3], DType::F32, &cpu()).unwrap();
    let err = map.get("nn_param", &[3], DType::BF16, &cpu()).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("nn_param"),
        "error should contain the variable name: {msg}"
    );
}

#[test]
fn test_varmap_shape_mismatch_error_contains_expected_and_got() {
    let mut map = VarMap::new();
    map.get("w", &[3, 4], DType::F32, &cpu()).unwrap();
    let err = map.get("w", &[5, 6], DType::F32, &cpu()).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("[3, 4]") && msg.contains("[5, 6]"),
        "error should show both shapes: {msg}"
    );
}
