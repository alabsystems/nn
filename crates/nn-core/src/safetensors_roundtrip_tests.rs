// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Safetensors serialization roundtrip tests.
//!
//! Exercises save/load paths for `save_safetensors`, `load_safetensors`,
//! `load_safetensors_from_bytes`, and `tensors_to_safetensors_bytes` across
//! dtypes, shapes, and edge cases.
//!
//! Part of #4186.

use std::collections::HashMap;

use crate::dyn_tensor::{
    load_safetensors, load_safetensors_from_bytes, save_safetensors, tensors_to_safetensors_bytes,
};
use crate::{DType, Device, DynTensor};

// ---------------------------------------------------------------------------
// A. Single tensor roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_single_f32_tensor_roundtrip_exact() {
    let values = vec![1.5, -2.25, 0.0, 3.75, -0.125];
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".to_string(),
        DynTensor::from_vec(values.clone(), &[5], &Device::Cpu).unwrap(),
    );
    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    assert_eq!(loaded.len(), 1);
    let t = &loaded["weight"];
    assert_eq!(t.dims(), &[5]);
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), values);
}

// ---------------------------------------------------------------------------
// B. Multiple tensors with different shapes
// ---------------------------------------------------------------------------

#[test]
fn test_multiple_tensors_different_shapes() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "scalar".to_string(),
        DynTensor::new(&[42.0], &[], &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "vector".to_string(),
        DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "matrix".to_string(),
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "rank3".to_string(),
        DynTensor::zeros(&[2, 3, 4], DType::F32, &Device::Cpu).unwrap(),
    );

    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();

    assert_eq!(loaded.len(), 4);
    assert_eq!(loaded["scalar"].to_flat_vec::<f32>().unwrap(), vec![42.0]);
    assert_eq!(loaded["vector"].dims(), &[3]);
    assert_eq!(loaded["matrix"].dims(), &[2, 3]);
    assert_eq!(loaded["rank3"].dims(), &[2, 3, 4]);
}

// ---------------------------------------------------------------------------
// C. BF16 native load (built from raw safetensors bytes)
// ---------------------------------------------------------------------------

/// Helper: build raw safetensors bytes with a single tensor of given dtype.
fn build_raw_safetensors(
    name: &str,
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    data: &[u8],
) -> Vec<u8> {
    let view = safetensors::tensor::TensorView::new(dtype, shape, data).unwrap();
    safetensors::tensor::serialize(vec![(name.to_string(), view)], None).unwrap()
}

#[test]
fn test_load_bf16_tensor_preserves_dtype() {
    let f32_values = [0.5f32, -1.0, 2.25, 3.0];
    let bf16_bytes: Vec<u8> = f32_values
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
        .collect();
    let bytes = build_raw_safetensors("w", safetensors::Dtype::BF16, vec![4], &bf16_bytes);
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    let t = &loaded["w"];
    assert_eq!(t.dtype(), DType::BF16);
    assert_eq!(t.dims(), &[4]);
    // Check values after converting back to f32.
    let f32_vals = t.to_f32_array().unwrap();
    let f32_vec: Vec<f32> = f32_vals.iter().copied().collect();
    for (got, expected) in f32_vec.iter().zip(f32_values.iter()) {
        assert!(
            (got - expected).abs() < 0.1,
            "BF16 roundtrip: got {got}, expected {expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// D. F16 native load
// ---------------------------------------------------------------------------

#[test]
fn test_load_f16_tensor_preserves_dtype() {
    let f32_values = [1.0f32, -0.5, 0.25, 7.5];
    let f16_bytes: Vec<u8> = f32_values
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect();
    let bytes = build_raw_safetensors("w", safetensors::Dtype::F16, vec![2, 2], &f16_bytes);
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    let t = &loaded["w"];
    assert_eq!(t.dtype(), DType::F16);
    assert_eq!(t.dims(), &[2, 2]);
    let f32_vals = t.to_f32_array().unwrap();
    let f32_vec: Vec<f32> = f32_vals.iter().copied().collect();
    for (got, expected) in f32_vec.iter().zip(f32_values.iter()) {
        assert!(
            (got - expected).abs() < 0.01,
            "F16 roundtrip: got {got}, expected {expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// E. F32 save roundtrip converts all float types to F32
// ---------------------------------------------------------------------------

#[test]
fn test_save_roundtrip_preserves_f32_values_exactly() {
    let dir = std::env::temp_dir().join(format!("nn_st_rt_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("roundtrip.safetensors");

    let values = vec![-1.0, 0.0, 1.0, f32::MIN_POSITIVE, f32::MAX / 2.0];
    let mut tensors = HashMap::new();
    tensors.insert(
        "exact".to_string(),
        DynTensor::from_vec(values.clone(), &[5], &Device::Cpu).unwrap(),
    );
    save_safetensors(&tensors, &path).unwrap();
    let loaded = load_safetensors(&path).unwrap();
    assert_eq!(loaded["exact"].to_flat_vec::<f32>().unwrap(), values);
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// F. Large tensor roundtrip (1M elements)
// ---------------------------------------------------------------------------

#[test]
fn test_large_tensor_roundtrip_1m_elements() {
    let n = 1_000_000;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let mut tensors = HashMap::new();
    tensors.insert(
        "large".to_string(),
        DynTensor::from_vec(data, &[1000, 1000], &Device::Cpu).unwrap(),
    );
    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    let t = &loaded["large"];
    assert_eq!(t.dims(), &[1000, 1000]);
    let loaded_data = t.to_flat_vec::<f32>().unwrap();
    assert_eq!(loaded_data.len(), n);
    // Check first, middle, and last values.
    assert!((loaded_data[0] - 0.0).abs() < 1e-7);
    assert!((loaded_data[500_000] - 500.0).abs() < 1e-3);
    assert!((loaded_data[999_999] - 999.999).abs() < 1e-3);
}

// ---------------------------------------------------------------------------
// G. Tensor name preservation
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_names_preserved_through_roundtrip() {
    let names = [
        "model.encoder.layers.0.self_attn.q_proj.weight",
        "model.encoder.layers.0.self_attn.k_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
    ];
    let mut tensors = HashMap::new();
    for (i, name) in names.iter().enumerate() {
        tensors.insert(
            name.to_string(),
            DynTensor::from_vec(vec![(i + 1) as f32], &[1], &Device::Cpu).unwrap(),
        );
    }

    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();

    assert_eq!(loaded.len(), names.len());
    for name in &names {
        assert!(loaded.contains_key(*name), "missing tensor name: {name}");
    }
    // Verify values map to correct names.
    for (i, name) in names.iter().enumerate() {
        assert_eq!(
            loaded[*name].to_flat_vec::<f32>().unwrap(),
            vec![(i + 1) as f32],
            "wrong value for tensor {name}"
        );
    }
}

// ---------------------------------------------------------------------------
// H. Empty tensor map
// ---------------------------------------------------------------------------

#[test]
fn test_empty_tensor_map_roundtrip() {
    let tensors: HashMap<String, DynTensor> = HashMap::new();
    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();
    assert!(
        !bytes.is_empty(),
        "even empty safetensors should have a header"
    );
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    assert!(loaded.is_empty());
}

// ---------------------------------------------------------------------------
// I. Byte-level serialization produces valid safetensors
// ---------------------------------------------------------------------------

#[test]
fn test_tensors_to_bytes_produces_parseable_output() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "a".to_string(),
        DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "b".to_string(),
        DynTensor::from_vec(vec![3.0], &[1], &Device::Cpu).unwrap(),
    );
    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();

    // Verify the bytes parse as valid safetensors.
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let tensor_names: Vec<String> = st.tensors().into_iter().map(|(name, _view)| name).collect();
    assert!(tensor_names.contains(&"a".to_string()));
    assert!(tensor_names.contains(&"b".to_string()));
}

// ---------------------------------------------------------------------------
// J. Unsupported dtype returns error
// ---------------------------------------------------------------------------

#[test]
fn test_load_unsupported_dtype_u8_returns_error() {
    let data = vec![0u8, 1, 2, 3];
    let bytes = build_raw_safetensors("x", safetensors::Dtype::U8, vec![4], &data);
    let err = load_safetensors_from_bytes(&bytes).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unsupported"),
        "expected unsupported dtype error, got: {msg}"
    );
}

#[test]
fn test_load_unsupported_dtype_i64_returns_error() {
    let data: Vec<u8> = [42i64, -1]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let bytes = build_raw_safetensors("x", safetensors::Dtype::I64, vec![2], &data);
    let err = load_safetensors_from_bytes(&bytes).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unsupported"),
        "expected unsupported dtype error, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// K. File I/O roundtrip with multiple tensors
// ---------------------------------------------------------------------------

#[test]
fn test_file_roundtrip_multiple_tensors() {
    let dir = std::env::temp_dir().join(format!("nn_st_multi_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("multi.safetensors");

    let mut tensors = HashMap::new();
    tensors.insert(
        "encoder.weight".to_string(),
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "encoder.bias".to_string(),
        DynTensor::from_vec(vec![0.1, 0.2], &[2], &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "decoder.weight".to_string(),
        DynTensor::from_vec(vec![5.0, 6.0, 7.0], &[3], &Device::Cpu).unwrap(),
    );

    save_safetensors(&tensors, &path).unwrap();
    assert!(path.exists());
    let loaded = load_safetensors(&path).unwrap();

    assert_eq!(loaded.len(), 3);
    assert_eq!(
        loaded["encoder.weight"].to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        loaded["encoder.bias"].to_flat_vec::<f32>().unwrap(),
        vec![0.1, 0.2]
    );
    assert_eq!(
        loaded["decoder.weight"].to_flat_vec::<f32>().unwrap(),
        vec![5.0, 6.0, 7.0]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// L. Mixed BF16 and F16 in same file
// ---------------------------------------------------------------------------

#[test]
fn test_load_mixed_bf16_and_f16_same_file() {
    // Build a safetensors file with one BF16 and one F16 tensor.
    let bf16_vals = [1.0f32, -2.0];
    let bf16_bytes: Vec<u8> = bf16_vals
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
        .collect();
    let f16_vals = [3.0f32, 4.0];
    let f16_bytes: Vec<u8> = f16_vals
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect();

    let bf16_view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![2], &bf16_bytes)
            .unwrap();
    let f16_view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![2], &f16_bytes).unwrap();
    let bytes = safetensors::tensor::serialize(
        vec![("bf".to_string(), bf16_view), ("fp".to_string(), f16_view)],
        None,
    )
    .unwrap();

    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded["bf"].dtype(), DType::BF16);
    assert_eq!(loaded["fp"].dtype(), DType::F16);
}

// ---------------------------------------------------------------------------
// M. Special float values (zero, negative zero, very small)
// ---------------------------------------------------------------------------

#[test]
fn test_special_float_values_roundtrip() {
    let values = vec![
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1e-38,
        -1e-38,
    ];
    let mut tensors = HashMap::new();
    tensors.insert(
        "special".to_string(),
        DynTensor::from_vec(values.clone(), &[6], &Device::Cpu).unwrap(),
    );
    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    let loaded_vals = loaded["special"].to_flat_vec::<f32>().unwrap();
    for (i, (got, expected)) in loaded_vals.iter().zip(values.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "bit-exact mismatch at index {i}: got {got}, expected {expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// N. High-rank tensor roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_high_rank_tensor_roundtrip() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "rank5".to_string(),
        DynTensor::zeros(&[2, 3, 4, 5, 6], DType::F32, &Device::Cpu).unwrap(),
    );
    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    assert_eq!(loaded["rank5"].dims(), &[2, 3, 4, 5, 6]);
    assert_eq!(loaded["rank5"].numel(), 2 * 3 * 4 * 5 * 6);
}
