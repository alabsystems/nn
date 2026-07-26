#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SafeTensorsBackend (D5 Direction 2 of #914, #915).

use std::io::Write;
use std::path::Path;

use nn_core::var_builder::TensorBackend;
use nn_core::{DType, Device};

use crate::context::MetalContext;
use crate::safetensors::WeightMap;
use crate::var_builder_safetensors::{var_builder_from_weight_map, SafeTensorsBackend};

// -- Helpers ------------------------------------------------------------------

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("nn_vb_st_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Create a safetensors file with a single f32 tensor.
fn create_single_tensor_file(path: &Path, name: &str, shape: &[usize], values: &[f32]) {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype as StDtype;

    let bytes = bytemuck::cast_slice::<f32, u8>(values);
    let view = TensorView::new(StDtype::F32, shape.to_vec(), bytes).expect("valid view");
    let tensors = vec![(name.to_string(), view)];
    let serialized = serialize(tensors, None).expect("serialize");
    let mut file = std::fs::File::create(path).expect("create file");
    file.write_all(&serialized).expect("write file");
}

/// Create a safetensors file with multiple named f32 tensors.
fn create_multi_tensor_file(path: &Path, tensors: &[(&str, &[usize], &[f32])]) {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype as StDtype;

    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|&(name, shape, values)| {
            let bytes = bytemuck::cast_slice::<f32, u8>(values);
            let view = TensorView::new(StDtype::F32, shape.to_vec(), bytes).expect("valid view");
            (name.to_string(), view)
        })
        .collect();
    let serialized = serialize(views, None).expect("serialize");
    std::fs::write(path, &serialized).expect("write file");
}

/// Create a safetensors file with a bf16 tensor containing known values.
fn create_bf16_tensor_file(path: &Path, name: &str, values: &[f32]) {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype as StDtype;

    let bf16_bytes: Vec<u8> = values
        .iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
        .collect();
    let view = TensorView::new(StDtype::BF16, vec![values.len()], &bf16_bytes).expect("valid view");
    let tensors = vec![(name.to_string(), view)];
    let serialized = serialize(tensors, None).expect("serialize");
    std::fs::write(path, &serialized).expect("write file");
}

/// Create a safetensors file with an f16 tensor containing known values.
fn create_f16_tensor_file(path: &Path, name: &str, values: &[f32]) {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype as StDtype;

    let f16_bytes: Vec<u8> = values
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect();
    let view = TensorView::new(StDtype::F16, vec![values.len()], &f16_bytes).expect("valid view");
    let tensors = vec![(name.to_string(), view)];
    let serialized = serialize(tensors, None).expect("serialize");
    std::fs::write(path, &serialized).expect("write file");
}

fn load_weight_map(path: &Path) -> WeightMap {
    let ctx = MetalContext::new().expect("Metal context");
    // SAFETY: Test file is not modified during the test.
    unsafe { WeightMap::load(path, &ctx).expect("load weight map") }
}

// -- get() tests --------------------------------------------------------------

#[test]
fn test_get_returns_correct_tensor() {
    let dir = temp_dir("get_correct");
    let path = dir.join("model.safetensors");
    create_single_tensor_file(&path, "weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let wm = load_weight_map(&path);
    let backend = SafeTensorsBackend::new(wm);

    let t = backend
        .get(&[2, 3], "weight", DType::F32, &Device::Cpu)
        .expect("get should succeed");
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.dtype(), DType::F32);
    let data = t.to_flat_vec::<f32>().expect("readback");
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_get_shape_mismatch() {
    let dir = temp_dir("get_shape_mismatch");
    let path = dir.join("model.safetensors");
    create_single_tensor_file(&path, "weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let wm = load_weight_map(&path);
    let backend = SafeTensorsBackend::new(wm);

    let err = backend.get(&[3, 2], "weight", DType::F32, &Device::Cpu);
    assert!(err.is_err(), "shape mismatch should return error");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_get_tensor_not_found() {
    let dir = temp_dir("get_not_found");
    let path = dir.join("model.safetensors");
    create_single_tensor_file(&path, "weight", &[3], &[1.0, 2.0, 3.0]);

    let wm = load_weight_map(&path);
    let backend = SafeTensorsBackend::new(wm);

    let err = backend.get(&[3], "nonexistent", DType::F32, &Device::Cpu);
    assert!(err.is_err(), "missing tensor should return error");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_get_bf16_converts_to_f32() {
    let dir = temp_dir("get_bf16_convert");
    let path = dir.join("model.safetensors");
    let values = [1.0f32, 2.5, -3.0, 0.0];
    create_bf16_tensor_file(&path, "weight", &values);

    let wm = load_weight_map(&path);
    let backend = SafeTensorsBackend::new(wm);

    let t = backend
        .get(&[4], "weight", DType::F32, &Device::Cpu)
        .expect("bf16 should auto-convert to f32");
    assert_eq!(t.dims(), &[4]);
    assert_eq!(t.dtype(), DType::F32);
    let data = t.to_flat_vec::<f32>().expect("readback");
    // bf16 has limited precision — check within tolerance
    for (got, expected) in data.iter().zip(values.iter()) {
        assert!(
            (got - expected).abs() < 0.02,
            "bf16 conversion: got {got}, expected {expected}"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_get_f16_converts_to_f32() {
    let dir = temp_dir("get_f16_convert");
    let path = dir.join("model.safetensors");
    let values = [1.0f32, -0.5, 3.125, 0.0];
    create_f16_tensor_file(&path, "weight", &values);

    let wm = load_weight_map(&path);
    let backend = SafeTensorsBackend::new(wm);

    let t = backend
        .get(&[4], "weight", DType::F32, &Device::Cpu)
        .expect("f16 should auto-convert to f32");
    assert_eq!(t.dims(), &[4]);
    assert_eq!(t.dtype(), DType::F32);
    let data = t.to_flat_vec::<f32>().expect("readback");
    // f16 has ~3.3 decimal digits precision — check within tolerance
    for (got, expected) in data.iter().zip(values.iter()) {
        assert!(
            (got - expected).abs() < 0.005,
            "f16 conversion: got {got}, expected {expected}"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_get_bf16_nan_rejected() {
    let dir = temp_dir("get_bf16_nan");
    let path = dir.join("model.safetensors");
    // Include a NaN value — should be rejected by defense-in-depth check
    create_bf16_tensor_file(&path, "weight", &[1.0, f32::NAN, 3.0]);

    let wm = load_weight_map(&path);
    let backend = SafeTensorsBackend::new(wm);

    let err = backend.get(&[3], "weight", DType::F32, &Device::Cpu);
    assert!(err.is_err(), "NaN in bf16 data should be rejected");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_get_unchecked_bf16_converts() {
    let dir = temp_dir("unchecked_bf16");
    let path = dir.join("model.safetensors");
    create_bf16_tensor_file(&path, "bias", &[0.1, 0.2, 0.3]);

    let wm = load_weight_map(&path);
    let backend = SafeTensorsBackend::new(wm);

    let t = backend
        .get_unchecked("bias", DType::F32, &Device::Cpu)
        .expect("bf16 get_unchecked should auto-convert");
    assert_eq!(t.dims(), &[3]);
    let data = t.to_flat_vec::<f32>().expect("readback");
    for (got, expected) in data.iter().zip([0.1f32, 0.2, 0.3].iter()) {
        assert!(
            (got - expected).abs() < 0.02,
            "bf16 unchecked: got {got}, expected {expected}"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

// -- get_unchecked() tests ----------------------------------------------------

#[test]
fn test_get_unchecked_returns_correct_tensor() {
    let dir = temp_dir("get_unchecked");
    let path = dir.join("model.safetensors");
    create_single_tensor_file(&path, "bias", &[4], &[0.1, 0.2, 0.3, 0.4]);

    let wm = load_weight_map(&path);
    let backend = SafeTensorsBackend::new(wm);

    let t = backend
        .get_unchecked("bias", DType::F32, &Device::Cpu)
        .expect("get_unchecked should succeed");
    assert_eq!(t.dims(), &[4]);
    let data = t.to_flat_vec::<f32>().expect("readback");
    assert_eq!(data, vec![0.1, 0.2, 0.3, 0.4]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_get_unchecked_not_found() {
    let dir = temp_dir("unchecked_not_found");
    let path = dir.join("model.safetensors");
    create_single_tensor_file(&path, "weight", &[2], &[1.0, 2.0]);

    let wm = load_weight_map(&path);
    let backend = SafeTensorsBackend::new(wm);

    let err = backend.get_unchecked("missing", DType::F32, &Device::Cpu);
    assert!(err.is_err(), "missing tensor should return error");

    std::fs::remove_dir_all(&dir).ok();
}

// -- contains_tensor() tests --------------------------------------------------

#[test]
fn test_contains_tensor_present() {
    let dir = temp_dir("contains_present");
    let path = dir.join("model.safetensors");
    create_multi_tensor_file(
        &path,
        &[
            ("encoder.weight", &[3], &[1.0, 2.0, 3.0]),
            ("encoder.bias", &[2], &[0.1, 0.2]),
        ],
    );

    let wm = load_weight_map(&path);
    let backend = SafeTensorsBackend::new(wm);

    assert!(backend.contains_tensor("encoder.weight"));
    assert!(backend.contains_tensor("encoder.bias"));
    assert!(!backend.contains_tensor("decoder.weight"));

    std::fs::remove_dir_all(&dir).ok();
}

// -- VarBuilder integration tests ---------------------------------------------

#[test]
fn test_var_builder_from_weight_map() {
    let dir = temp_dir("vb_from_wm");
    let path = dir.join("model.safetensors");
    create_multi_tensor_file(
        &path,
        &[
            (
                "encoder.conv.weight",
                &[2, 3],
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ),
            ("encoder.conv.bias", &[2], &[0.5, 0.6]),
            ("decoder.weight", &[4], &[7.0, 8.0, 9.0, 10.0]),
        ],
    );

    let wm = load_weight_map(&path);
    let vb = var_builder_from_weight_map(wm, DType::F32, &Device::Cpu);

    // Test pp() scoping with get()
    let enc = vb.pp("encoder").pp("conv");
    let w = enc.get(&[2, 3], "weight").expect("load weight");
    assert_eq!(w.dims(), &[2, 3]);
    let data = w.to_flat_vec::<f32>().expect("readback");
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Test bias
    let b = enc.get(&[2], "bias").expect("load bias");
    let bias_data = b.to_flat_vec::<f32>().expect("readback");
    assert_eq!(bias_data, vec![0.5, 0.6]);

    // Test direct access (no pp)
    let dec = vb.get(&[4], "decoder.weight").expect("load decoder weight");
    let dec_data = dec.to_flat_vec::<f32>().expect("readback");
    assert_eq!(dec_data, vec![7.0, 8.0, 9.0, 10.0]);

    // Test contains_tensor via VarBuilder
    assert!(vb.pp("encoder").pp("conv").contains_tensor("weight"));
    assert!(!vb.pp("encoder").contains_tensor("nonexistent"));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_var_builder_shape_error_via_vb() {
    let dir = temp_dir("vb_shape_err");
    let path = dir.join("model.safetensors");
    create_single_tensor_file(&path, "w", &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let wm = load_weight_map(&path);
    let vb = var_builder_from_weight_map(wm, DType::F32, &Device::Cpu);

    let err = vb.get(&[2, 3], "w"); // wrong shape
    assert!(
        err.is_err(),
        "shape mismatch through VarBuilder should fail"
    );

    std::fs::remove_dir_all(&dir).ok();
}

// -- E2E / mmap tests extracted to var_builder_safetensors_e2e_tests.rs -------

#[cfg(test)]
#[path = "var_builder_safetensors_e2e_tests.rs"]
mod e2e_tests;
