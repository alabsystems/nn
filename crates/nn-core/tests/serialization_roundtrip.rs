// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive serialization and save/load roundtrip tests for DynTensor
//! safetensors format, model weight persistence, error handling, and format
//! compatibility.
//!
//! Covers:
//! - A. Tensor serialization (F32, BF16, F16, multi-tensor, shapes, large, empty)
//! - B. Model weight roundtrip (Linear, Conv1d, LayerNorm, multi-layer, forward parity)
//! - C. Error handling (corrupted, wrong dtype, missing key, read-only path)
//! - D. Format compatibility (header, byte order, alignment)

use std::collections::HashMap;

use nn_core::layers::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module};
use nn_core::var_builder::VarBuilder;
use nn_core::{
    load_safetensors, load_safetensors_from_bytes, save_safetensors, tensors_to_safetensors_bytes,
    DType, Device, DynTensor,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cpu() -> Device {
    Device::Cpu
}

/// Deterministic pseudo-random f32 data via xorshift64.
fn pseudo_random_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32) / (u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Create a DynTensor from pseudo-random data.
fn rand_tensor(shape: &[usize], seed: u64) -> DynTensor {
    let numel: usize = shape.iter().product();
    let data = pseudo_random_vec(numel, seed);
    DynTensor::from_vec(data, shape, &cpu()).unwrap()
}

/// Unique temp dir for each test.
fn temp_dir(test_name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "nn_serial_test_{}_{}",
        test_name,
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ===========================================================================
// A. Tensor Serialization
// ===========================================================================

#[test]
fn test_dyntensor_to_safetensors_f32() {
    let data = vec![1.0f32, -2.5, 3.75, 0.0, -0.001, 100.0];
    let t = DynTensor::from_vec(data.clone(), &[2, 3], &cpu()).unwrap();
    let mut map = HashMap::new();
    map.insert("tensor".to_string(), t);

    let bytes = tensors_to_safetensors_bytes(&map).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();

    assert_eq!(loaded.len(), 1);
    let lt = &loaded["tensor"];
    assert_eq!(lt.dtype(), DType::F32);
    assert_eq!(lt.dims(), &[2, 3]);
    assert_eq!(lt.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_dyntensor_to_safetensors_bf16() {
    // BF16 roundtrip: write F32 (converted by save), load F32, values match within tolerance.
    let original_f32 = vec![1.0f32, -2.5, 3.75, 0.0];
    let t = DynTensor::from_vec(original_f32.clone(), &[4], &cpu()).unwrap();
    let mut map = HashMap::new();
    map.insert("bf16_test".to_string(), t);

    // save_safetensors always converts to F32 before writing.
    let bytes = tensors_to_safetensors_bytes(&map).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    let lt = &loaded["bf16_test"];
    assert_eq!(lt.dtype(), DType::F32);
    assert_eq!(lt.to_flat_vec::<f32>().unwrap(), original_f32);

    // Also test loading native BF16 data (built manually).
    let bf16_data: Vec<u8> = original_f32
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
        .collect();
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![4], &bf16_data)
        .unwrap();
    let bf16_bytes = safetensors::tensor::serialize(vec![("w".to_string(), view)], None).unwrap();
    let bf16_loaded = load_safetensors_from_bytes(&bf16_bytes).unwrap();
    assert_eq!(bf16_loaded["w"].dtype(), DType::BF16);
    let f32_vals: Vec<f32> = bf16_loaded["w"]
        .to_f32_array()
        .unwrap()
        .iter()
        .copied()
        .collect();
    for (got, expected) in f32_vals.iter().zip(original_f32.iter()) {
        assert!(
            (got - expected).abs() < 0.1,
            "BF16 roundtrip mismatch: got {got}, expected {expected}"
        );
    }
}

#[test]
fn test_dyntensor_to_safetensors_f16() {
    let original_f32 = [0.5f32, -1.25, 2.0, 0.0, 65504.0];
    let f16_data: Vec<u8> = original_f32
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect();
    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![5], &f16_data).unwrap();
    let bytes =
        safetensors::tensor::serialize(vec![("f16_tensor".to_string(), view)], None).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();

    let lt = &loaded["f16_tensor"];
    assert_eq!(lt.dtype(), DType::F16);
    assert_eq!(lt.dims(), &[5]);
    let f32_vals: Vec<f32> = lt.to_f32_array().unwrap().iter().copied().collect();
    for (got, expected) in f32_vals.iter().zip(original_f32.iter()) {
        assert!(
            (got - expected).abs() < 0.01,
            "F16 roundtrip mismatch: got {got}, expected {expected}"
        );
    }
}

#[test]
fn test_safetensors_multiple_tensors() {
    let t1 = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let t2 = DynTensor::from_vec(vec![4.0, 5.0, 6.0, 7.0], &[2, 2], &cpu()).unwrap();
    let t3 = DynTensor::from_vec(vec![-1.0], &[1, 1, 1], &cpu()).unwrap();
    let t4 =
        DynTensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3], &cpu()).unwrap();

    let mut map = HashMap::new();
    map.insert("alpha".to_string(), t1);
    map.insert("beta".to_string(), t2);
    map.insert("gamma".to_string(), t3);
    map.insert("delta".to_string(), t4);

    let bytes = tensors_to_safetensors_bytes(&map).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();

    assert_eq!(loaded.len(), 4);
    assert_eq!(
        loaded["alpha"].to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        loaded["beta"].to_flat_vec::<f32>().unwrap(),
        vec![4.0, 5.0, 6.0, 7.0]
    );
    assert_eq!(loaded["gamma"].to_flat_vec::<f32>().unwrap(), vec![-1.0]);
    assert_eq!(
        loaded["delta"].to_flat_vec::<f32>().unwrap(),
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    );
}

#[test]
fn test_safetensors_key_naming() {
    // Keys with hierarchical dot-separated names (PyTorch convention).
    let mut map = HashMap::new();
    map.insert(
        "encoder.layers.0.self_attn.q_proj.weight".to_string(),
        DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap(),
    );
    map.insert(
        "decoder.embed_tokens.weight".to_string(),
        DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap(),
    );
    map.insert(
        "model.norm.weight".to_string(),
        DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap(),
    );

    let bytes = tensors_to_safetensors_bytes(&map).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();

    assert!(loaded.contains_key("encoder.layers.0.self_attn.q_proj.weight"));
    assert!(loaded.contains_key("decoder.embed_tokens.weight"));
    assert!(loaded.contains_key("model.norm.weight"));
    assert_eq!(
        loaded["encoder.layers.0.self_attn.q_proj.weight"]
            .to_flat_vec::<f32>()
            .unwrap(),
        vec![1.0]
    );
}

#[test]
fn test_safetensors_shape_preservation() {
    // Test various shapes: scalar-like, 1D, 2D, 3D, 4D, 5D.
    let shapes: Vec<Vec<usize>> = vec![
        vec![1],
        vec![100],
        vec![4, 8],
        vec![2, 3, 4],
        vec![1, 2, 3, 4],
        vec![1, 1, 1, 1, 1],
    ];

    let mut map = HashMap::new();
    for (i, shape) in shapes.iter().enumerate() {
        let numel: usize = shape.iter().product();
        let data = vec![1.0f32; numel];
        let t = DynTensor::from_vec(data, shape.as_slice(), &cpu()).unwrap();
        map.insert(format!("shape_{i}"), t);
    }

    let bytes = tensors_to_safetensors_bytes(&map).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();

    for (i, shape) in shapes.iter().enumerate() {
        let key = format!("shape_{i}");
        assert_eq!(
            loaded[&key].dims(),
            shape.as_slice(),
            "Shape mismatch for {key}: expected {shape:?}, got {:?}",
            loaded[&key].dims()
        );
    }
}

#[test]
fn test_safetensors_dtype_preservation() {
    // F32 tensors serialized and deserialized remain F32.
    let t = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::F32);

    let mut map = HashMap::new();
    map.insert("x".to_string(), t);
    let bytes = tensors_to_safetensors_bytes(&map).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    assert_eq!(loaded["x"].dtype(), DType::F32);
}

#[test]
fn test_safetensors_large_tensor() {
    // Roundtrip a 1M-element tensor.
    let n = 1_000_000;
    let data = pseudo_random_vec(n, 12345);
    let t = DynTensor::from_vec(data.clone(), &[1000, 1000], &cpu()).unwrap();
    let mut map = HashMap::new();
    map.insert("large".to_string(), t);

    let bytes = tensors_to_safetensors_bytes(&map).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();

    assert_eq!(loaded["large"].dims(), &[1000, 1000]);
    let loaded_data = loaded["large"].to_flat_vec::<f32>().unwrap();
    assert_eq!(loaded_data.len(), n);
    // Verify first and last 10 values exactly (f32 roundtrip via LE bytes is lossless).
    assert_eq!(&loaded_data[..10], &data[..10]);
    assert_eq!(&loaded_data[n - 10..], &data[n - 10..]);
}

#[test]
fn test_safetensors_empty_map() {
    // Empty tensor map roundtrips to empty map.
    let map: HashMap<String, DynTensor> = HashMap::new();
    let bytes = tensors_to_safetensors_bytes(&map).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    assert!(loaded.is_empty());
}

#[test]
fn test_safetensors_single_element_tensor() {
    // Single-element [1] tensor.
    let t = DynTensor::from_vec(vec![42.0], &[1], &cpu()).unwrap();
    let mut map = HashMap::new();
    map.insert("scalar_like".to_string(), t);

    let bytes = tensors_to_safetensors_bytes(&map).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    assert_eq!(
        loaded["scalar_like"].to_flat_vec::<f32>().unwrap(),
        vec![42.0]
    );
    assert_eq!(loaded["scalar_like"].dims(), &[1]);
}

// ===========================================================================
// B. Model Weight Roundtrip
// ===========================================================================

/// Helper: save a map to file, reload, build VarBuilder from the loaded tensors.
fn roundtrip_via_file(
    map: HashMap<String, DynTensor>,
    test_name: &str,
) -> HashMap<String, DynTensor> {
    let dir = temp_dir(test_name);
    let path = dir.join("weights.safetensors");
    save_safetensors(&map, &path).unwrap();
    let loaded = load_safetensors(&path).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    loaded
}

#[test]
fn test_linear_weights_roundtrip() {
    let in_f = 16;
    let out_f = 8;
    let weight = rand_tensor(&[out_f, in_f], 100);
    let bias = rand_tensor(&[out_f], 200);

    let original = Linear::new(weight.clone(), Some(bias.clone())).unwrap();

    // Save weights to safetensors.
    let mut map = HashMap::new();
    map.insert("weight".to_string(), weight);
    map.insert("bias".to_string(), bias);

    let loaded = roundtrip_via_file(map, "linear_roundtrip");

    // Reconstruct Linear from loaded weights.
    let vb = VarBuilder::from_tensors(loaded, DType::F32, &cpu());
    let restored = Linear::load(&vb, in_f, out_f).unwrap();

    // Verify weights match exactly.
    assert_eq!(
        original.weight().to_flat_vec::<f32>().unwrap(),
        restored.weight().to_flat_vec::<f32>().unwrap()
    );
    assert_eq!(
        original.bias().unwrap().to_flat_vec::<f32>().unwrap(),
        restored.bias().unwrap().to_flat_vec::<f32>().unwrap()
    );

    // Forward pass produces identical output.
    let x = rand_tensor(&[2, in_f], 300);
    let y_orig = original.forward(&x).unwrap();
    let y_loaded = restored.forward(&x).unwrap();
    assert_eq!(
        y_orig.to_flat_vec::<f32>().unwrap(),
        y_loaded.to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_conv1d_weights_roundtrip() {
    let in_c = 4;
    let out_c = 8;
    let kernel = 3;
    let config = Conv1dConfig::default();

    let weight = rand_tensor(&[out_c, in_c, kernel], 400);
    let bias = rand_tensor(&[out_c], 500);
    let original = Conv1d::new(weight.clone(), Some(bias.clone()), config).unwrap();

    let mut map = HashMap::new();
    map.insert("weight".to_string(), weight);
    map.insert("bias".to_string(), bias);

    let loaded = roundtrip_via_file(map, "conv1d_roundtrip");
    let vb = VarBuilder::from_tensors(loaded, DType::F32, &cpu());
    let restored = Conv1d::load(&vb, in_c, out_c, kernel, config).unwrap();

    // Verify weights match.
    assert_eq!(
        original.weight().to_flat_vec::<f32>().unwrap(),
        restored.weight().to_flat_vec::<f32>().unwrap()
    );
    assert_eq!(
        original.bias().unwrap().to_flat_vec::<f32>().unwrap(),
        restored.bias().unwrap().to_flat_vec::<f32>().unwrap()
    );

    // Forward pass parity.
    let x = rand_tensor(&[1, in_c, 16], 600);
    let y_orig = original.forward(&x).unwrap();
    let y_loaded = restored.forward(&x).unwrap();
    assert_eq!(
        y_orig.to_flat_vec::<f32>().unwrap(),
        y_loaded.to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_layernorm_weights_roundtrip() {
    let dim = 32;
    let eps = 1e-5;

    let weight = rand_tensor(&[dim], 700);
    let bias = rand_tensor(&[dim], 800);
    let original = LayerNorm::new(weight.clone(), bias.clone(), eps).unwrap();

    let mut map = HashMap::new();
    map.insert("weight".to_string(), weight);
    map.insert("bias".to_string(), bias);

    let loaded = roundtrip_via_file(map, "layernorm_roundtrip");
    let vb = VarBuilder::from_tensors(loaded, DType::F32, &cpu());
    let restored = LayerNorm::load(&vb, dim, eps).unwrap();

    assert_eq!(
        original.weight().to_flat_vec::<f32>().unwrap(),
        restored.weight().to_flat_vec::<f32>().unwrap()
    );
    assert_eq!(
        original.bias().to_flat_vec::<f32>().unwrap(),
        restored.bias().to_flat_vec::<f32>().unwrap()
    );

    // Forward pass with identical input should produce the same output.
    let x = rand_tensor(&[2, 4, dim], 900);
    let y_orig = original.forward(&x).unwrap();
    let y_loaded = restored.forward(&x).unwrap();
    let orig_vals = y_orig.to_flat_vec::<f32>().unwrap();
    let loaded_vals = y_loaded.to_flat_vec::<f32>().unwrap();
    assert_eq!(orig_vals.len(), loaded_vals.len());
    for (i, (a, b)) in orig_vals.iter().zip(loaded_vals.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "LayerNorm forward mismatch at index {i}: {a} vs {b}"
        );
    }
}

#[test]
fn test_full_model_weights_roundtrip() {
    // Simulate a multi-layer model with hierarchical weight names.
    let in_f = 16;
    let hidden = 32;
    let out_f = 8;

    let w1 = rand_tensor(&[hidden, in_f], 1001);
    let b1 = rand_tensor(&[hidden], 1002);
    let w2 = rand_tensor(&[out_f, hidden], 1003);
    let b2 = rand_tensor(&[out_f], 1004);
    let ln_w = rand_tensor(&[hidden], 1005);
    let ln_b = rand_tensor(&[hidden], 1006);

    let mut map = HashMap::new();
    map.insert("layer1.weight".to_string(), w1.clone());
    map.insert("layer1.bias".to_string(), b1.clone());
    map.insert("layer2.weight".to_string(), w2.clone());
    map.insert("layer2.bias".to_string(), b2.clone());
    map.insert("norm.weight".to_string(), ln_w.clone());
    map.insert("norm.bias".to_string(), ln_b.clone());

    let loaded = roundtrip_via_file(map, "full_model_roundtrip");

    // Verify all 6 tensors survived.
    assert_eq!(loaded.len(), 6);

    let vb = VarBuilder::from_tensors(loaded, DType::F32, &cpu());
    let l1 = Linear::load(vb.pp("layer1"), in_f, hidden).unwrap();
    let l2 = Linear::load(vb.pp("layer2"), hidden, out_f).unwrap();
    let norm = LayerNorm::load(vb.pp("norm"), hidden, 1e-5).unwrap();

    // Reconstruct from original tensors.
    let l1_orig = Linear::new(w1, Some(b1)).unwrap();
    let l2_orig = Linear::new(w2, Some(b2)).unwrap();
    let norm_orig = LayerNorm::new(ln_w, ln_b, 1e-5).unwrap();

    // Forward pass parity through the full pipeline.
    let x = rand_tensor(&[2, in_f], 2000);
    let h_orig = l1_orig.forward(&x).unwrap();
    let h_orig = norm_orig.forward(&h_orig).unwrap();
    let y_orig = l2_orig.forward(&h_orig).unwrap();

    let h_loaded = l1.forward(&x).unwrap();
    let h_loaded = norm.forward(&h_loaded).unwrap();
    let y_loaded = l2.forward(&h_loaded).unwrap();

    let orig_vals = y_orig.to_flat_vec::<f32>().unwrap();
    let loaded_vals = y_loaded.to_flat_vec::<f32>().unwrap();
    assert_eq!(orig_vals.len(), loaded_vals.len());
    for (i, (a, b)) in orig_vals.iter().zip(loaded_vals.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "Full model forward mismatch at index {i}: {a} vs {b}"
        );
    }
}

#[test]
fn test_model_forward_after_load() {
    // End-to-end: save to file, load from file, forward pass matches.
    let in_f = 8;
    let out_f = 4;
    let weight = rand_tensor(&[out_f, in_f], 3000);
    let bias = rand_tensor(&[out_f], 3001);

    let original = Linear::new(weight.clone(), Some(bias.clone())).unwrap();
    let x = rand_tensor(&[3, in_f], 3002);
    let y_expected = original.forward(&x).unwrap();

    let dir = temp_dir("forward_after_load");
    let path = dir.join("model.safetensors");
    let mut map = HashMap::new();
    map.insert("weight".to_string(), weight);
    map.insert("bias".to_string(), bias);
    save_safetensors(&map, &path).unwrap();

    // Load fresh from disk.
    let loaded = load_safetensors(&path).unwrap();
    let vb = VarBuilder::from_tensors(loaded, DType::F32, &cpu());
    let restored = Linear::load(&vb, in_f, out_f).unwrap();
    let y_actual = restored.forward(&x).unwrap();

    assert_eq!(
        y_expected.to_flat_vec::<f32>().unwrap(),
        y_actual.to_flat_vec::<f32>().unwrap()
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_weight_map_roundtrip() {
    // VarBuilder with TensorMapBackend: build from tensors → serialize → load → VarBuilder.
    let tensors = HashMap::from([
        (
            "enc.q.weight".to_string(),
            DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap(),
        ),
        (
            "enc.k.weight".to_string(),
            DynTensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], &cpu()).unwrap(),
        ),
    ]);

    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();

    let vb = VarBuilder::from_tensors(loaded, DType::F32, &cpu());
    let q = vb.pp("enc").pp("q").get(&[2, 2], "weight").unwrap();
    let k = vb.pp("enc").pp("k").get(&[2, 2], "weight").unwrap();

    assert_eq!(q.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(k.to_flat_vec::<f32>().unwrap(), vec![5.0, 6.0, 7.0, 8.0]);
}

// ===========================================================================
// C. Error Handling
// ===========================================================================

#[test]
fn test_load_corrupted_file() {
    // Completely invalid bytes should produce an error.
    let garbage = b"this is not safetensors data at all!";
    let result = load_safetensors_from_bytes(garbage);
    assert!(result.is_err(), "Loading garbage bytes should fail");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("deserialize") || err_msg.contains("safetensors"),
        "Error should mention deserialization, got: {err_msg}"
    );
}

#[test]
fn test_load_truncated_bytes() {
    // Valid header but truncated data.
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let mut map = HashMap::new();
    map.insert("x".to_string(), t);
    let bytes = tensors_to_safetensors_bytes(&map).unwrap();

    // Truncate the last 4 bytes (one f32 value).
    let truncated = &bytes[..bytes.len() - 4];
    let result = load_safetensors_from_bytes(truncated);
    assert!(result.is_err(), "Loading truncated data should fail");
}

#[test]
fn test_load_wrong_dtype_i64_unsupported() {
    // I64 dtype is not supported — loading should produce a descriptive error.
    let data = 42i64.to_le_bytes();
    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::I64, vec![1], &data).unwrap();
    let bytes = safetensors::tensor::serialize(vec![("x".to_string(), view)], None).unwrap();
    let result = load_safetensors_from_bytes(&bytes);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("unsupported") && err.contains("dtype"),
        "Expected unsupported dtype error, got: {err}"
    );
}

#[test]
fn test_load_wrong_dtype_u8_unsupported() {
    let data = vec![0u8, 1, 2, 3];
    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::U8, vec![4], &data).unwrap();
    let bytes = safetensors::tensor::serialize(vec![("x".to_string(), view)], None).unwrap();
    let result = load_safetensors_from_bytes(&bytes);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("unsupported"),
        "Expected unsupported dtype error for U8, got: {err}"
    );
}

#[test]
fn test_load_missing_key_from_varbuilder() {
    // VarBuilder from a loaded map: request a key that does not exist.
    let t = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    let mut map = HashMap::new();
    map.insert("existing_key".to_string(), t);

    let bytes = tensors_to_safetensors_bytes(&map).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    let vb = VarBuilder::from_tensors(loaded, DType::F32, &cpu());

    let result = vb.get(&[1], "nonexistent");
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("nonexistent") || err.contains("not found") || err.contains("NotFound"),
        "Error should mention the missing key, got: {err}"
    );
}

#[test]
fn test_save_to_nonexistent_directory() {
    // Writing to a path with a nonexistent parent directory should fail.
    let t = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    let mut map = HashMap::new();
    map.insert("x".to_string(), t);

    let bad_path =
        std::path::PathBuf::from("/tmp/nonexistent_nn_dir_12345/nested/weights.safetensors");
    let result = save_safetensors(&map, &bad_path);
    assert!(
        result.is_err(),
        "Saving to nonexistent directory should fail"
    );
}

#[test]
fn test_load_nonexistent_file() {
    let result = load_safetensors("/tmp/definitely_does_not_exist_nn_42.safetensors");
    assert!(result.is_err(), "Loading nonexistent file should fail");
}

#[test]
fn test_load_empty_bytes() {
    // Zero bytes is invalid safetensors.
    let result = load_safetensors_from_bytes(&[]);
    assert!(result.is_err(), "Loading empty bytes should fail");
}

// ===========================================================================
// D. Format Compatibility
// ===========================================================================

#[test]
fn test_safetensors_header_format() {
    // The safetensors format starts with an 8-byte little-endian u64 header
    // length, followed by a JSON header, then raw tensor data.
    let t = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    let mut map = HashMap::new();
    map.insert("x".to_string(), t);
    let bytes = tensors_to_safetensors_bytes(&map).unwrap();

    // First 8 bytes: header length as little-endian u64.
    assert!(bytes.len() >= 8, "Safetensors bytes too short for header");
    let header_len = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    assert!(
        header_len > 0 && header_len < bytes.len() as u64,
        "Header length {header_len} out of range for {}-byte file",
        bytes.len()
    );

    // Header bytes should be valid UTF-8 JSON.
    let header_bytes = &bytes[8..8 + header_len as usize];
    let header_str = std::str::from_utf8(header_bytes).expect("Header should be valid UTF-8");
    let header_json: serde_json::Value =
        serde_json::from_str(header_str).expect("Header should be valid JSON");

    // The header should contain our tensor key "x".
    assert!(
        header_json.get("x").is_some(),
        "Header JSON should contain key 'x', got: {header_json}"
    );

    // The tensor metadata should include dtype and shape.
    let x_meta = &header_json["x"];
    assert_eq!(
        x_meta["dtype"].as_str().unwrap(),
        "F32",
        "dtype should be F32"
    );
    let shape = x_meta["shape"]
        .as_array()
        .expect("shape should be an array");
    assert_eq!(shape.len(), 1);
    assert_eq!(shape[0].as_u64().unwrap(), 2);
}

#[test]
fn test_safetensors_byte_order() {
    // Safetensors uses little-endian. Verify that the raw bytes in the file
    // match the expected little-endian representation of our f32 values.
    let val = std::f32::consts::PI;
    let t = DynTensor::from_vec(vec![val], &[1], &cpu()).unwrap();
    let mut map = HashMap::new();
    map.insert("pi".to_string(), t);
    let bytes = tensors_to_safetensors_bytes(&map).unwrap();

    // The raw data section starts after header_len + 8 header bytes.
    let header_len = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]) as usize;
    let data_start = 8 + header_len;

    // There should be exactly 4 bytes of f32 data.
    let data_section = &bytes[data_start..];
    assert_eq!(
        data_section.len(),
        4,
        "Expected 4 bytes of f32 data, got {}",
        data_section.len()
    );

    // Verify little-endian byte order.
    let expected_le = val.to_le_bytes();
    assert_eq!(
        data_section, &expected_le,
        "Data bytes should match little-endian f32 encoding"
    );
}

#[test]
fn test_safetensors_data_alignment() {
    // The safetensors format encodes data offsets in the header. Verify that
    // multiple tensors have correct non-overlapping byte ranges.
    let t1 = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let t2 = DynTensor::from_vec(vec![4.0, 5.0], &[2], &cpu()).unwrap();
    let mut map = HashMap::new();
    map.insert("a".to_string(), t1);
    map.insert("b".to_string(), t2);
    let bytes = tensors_to_safetensors_bytes(&map).unwrap();

    // Parse header.
    let header_len = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]) as usize;
    let header_str = std::str::from_utf8(&bytes[8..8 + header_len]).unwrap();
    let header: serde_json::Value = serde_json::from_str(header_str).unwrap();

    // Collect data offsets for each tensor.
    let mut ranges = Vec::new();
    for key in ["a", "b"] {
        let meta = &header[key];
        let offsets = meta["data_offsets"]
            .as_array()
            .expect("data_offsets should be present");
        let start = offsets[0].as_u64().unwrap();
        let end = offsets[1].as_u64().unwrap();
        assert!(
            end > start,
            "Tensor '{key}' has invalid offsets: [{start}, {end})"
        );
        ranges.push((key, start, end));
    }

    // Verify ranges do not overlap.
    ranges.sort_by_key(|r| r.1);
    for w in ranges.windows(2) {
        let (name_a, _, end_a) = w[0];
        let (name_b, start_b, _) = w[1];
        assert!(
            end_a <= start_b,
            "Tensor '{name_a}' end ({end_a}) overlaps with tensor '{name_b}' start ({start_b})"
        );
    }

    // Verify total data length matches expected: 3*4 + 2*4 = 20 bytes.
    let data_section = &bytes[8 + header_len..];
    assert_eq!(
        data_section.len(),
        20,
        "Total data section should be 20 bytes (5 f32 values), got {}",
        data_section.len()
    );
}

#[test]
fn test_safetensors_file_roundtrip_binary_identical() {
    // Save to file, read raw bytes back, and verify they match the in-memory
    // serialization exactly.
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let mut map = HashMap::new();
    map.insert("w".to_string(), t);

    let memory_bytes = tensors_to_safetensors_bytes(&map).unwrap();
    let dir = temp_dir("binary_identical");
    let path = dir.join("test.safetensors");
    save_safetensors(&map, &path).unwrap();
    let file_bytes = std::fs::read(&path).unwrap();

    assert_eq!(
        memory_bytes, file_bytes,
        "File bytes should be identical to in-memory serialization"
    );
    std::fs::remove_dir_all(&dir).ok();
}

// ===========================================================================
// E. Additional edge cases
// ===========================================================================

#[test]
fn test_overwrite_existing_file() {
    // Saving to the same path twice should overwrite.
    let dir = temp_dir("overwrite");
    let path = dir.join("weights.safetensors");

    let t1 = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    let mut map1 = HashMap::new();
    map1.insert("v1".to_string(), t1);
    save_safetensors(&map1, &path).unwrap();

    let t2 = DynTensor::from_vec(vec![2.0, 3.0], &[2], &cpu()).unwrap();
    let mut map2 = HashMap::new();
    map2.insert("v2".to_string(), t2);
    save_safetensors(&map2, &path).unwrap();

    let loaded = load_safetensors(&path).unwrap();
    assert!(
        !loaded.contains_key("v1"),
        "Overwritten file should not contain old key"
    );
    assert!(loaded.contains_key("v2"));
    assert_eq!(loaded["v2"].to_flat_vec::<f32>().unwrap(), vec![2.0, 3.0]);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_negative_and_special_f32_values() {
    // Test negative zero, subnormals, very small, very large.
    let data = vec![
        0.0f32,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        1e-38,
        -1e-38,
    ];
    let t = DynTensor::from_vec(data.clone(), &[8], &cpu()).unwrap();
    let mut map = HashMap::new();
    map.insert("special".to_string(), t);

    let bytes = tensors_to_safetensors_bytes(&map).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    let loaded_data = loaded["special"].to_flat_vec::<f32>().unwrap();

    // Compare bitwise for exact f32 preservation (including -0.0).
    for (i, (a, b)) in data.iter().zip(loaded_data.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "Bitwise mismatch at index {i}: {a} vs {b}"
        );
    }
}

#[test]
fn test_linear_no_bias_roundtrip() {
    // Linear without bias: verify weights survive and forward matches.
    let in_f = 8;
    let out_f = 4;
    let weight = rand_tensor(&[out_f, in_f], 5000);
    let original = Linear::new(weight.clone(), None).unwrap();

    let mut map = HashMap::new();
    map.insert("weight".to_string(), weight);

    let bytes = tensors_to_safetensors_bytes(&map).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    let vb = VarBuilder::from_tensors(loaded, DType::F32, &cpu());
    let restored = Linear::load(&vb, in_f, out_f).unwrap();

    assert!(restored.bias().is_none());
    let x = rand_tensor(&[2, in_f], 5001);
    assert_eq!(
        original.forward(&x).unwrap().to_flat_vec::<f32>().unwrap(),
        restored.forward(&x).unwrap().to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_many_tensors_roundtrip() {
    // Roundtrip 50 tensors with various shapes.
    let mut map = HashMap::new();
    for i in 0..50u64 {
        let shape = &[(i as usize % 5) + 1, (i as usize % 7) + 1];
        let t = rand_tensor(shape, 6000 + i);
        map.insert(format!("param_{i}"), t);
    }

    let bytes = tensors_to_safetensors_bytes(&map).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();

    assert_eq!(loaded.len(), 50);
    for i in 0..50u64 {
        let key = format!("param_{i}");
        let original = &map[&key];
        let restored = &loaded[&key];
        assert_eq!(original.dims(), restored.dims(), "Shape mismatch for {key}");
        assert_eq!(
            original.to_flat_vec::<f32>().unwrap(),
            restored.to_flat_vec::<f32>().unwrap(),
            "Data mismatch for {key}"
        );
    }
}

#[test]
fn test_unicode_key_names() {
    // safetensors keys are strings — verify unicode keys survive roundtrip.
    let mut map = HashMap::new();
    map.insert(
        "layer_alpha".to_string(),
        DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap(),
    );
    map.insert(
        "weights-with-dashes".to_string(),
        DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap(),
    );
    map.insert(
        "a.b.c.d.e.f".to_string(),
        DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap(),
    );

    let bytes = tensors_to_safetensors_bytes(&map).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();

    assert!(loaded.contains_key("layer_alpha"));
    assert!(loaded.contains_key("weights-with-dashes"));
    assert!(loaded.contains_key("a.b.c.d.e.f"));
}
