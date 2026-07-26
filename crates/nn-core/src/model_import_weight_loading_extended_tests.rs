// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended model import and weight loading tests.
//!
//! Covers areas not exercised by existing VarBuilder test files:
//! - Multi-file safetensors loading (sharded checkpoints)
//! - Shape validation for real model configurations (transformer, conv, etc.)
//! - Missing weight detection and graceful error reporting
//! - Extra/unused weight detection patterns
//! - Quantized weight loading (Q4_0, Q8_0 block quantization)
//! - Memory-mapped loading simulation (mmap pattern via file round-trip)
//! - PyTorch-to-nn naming convention mapping for real model families
//! - Weight dtype conversion during import (F32 <-> BF16/F16)
//! - Multi-layer transformer weight pattern loading
//!
//! Part of #4560.

use std::collections::HashMap;
use std::sync::Arc;

use crate::dyn_tensor::{
    load_safetensors, load_safetensors_from_bytes, save_safetensors, tensors_to_safetensors_bytes,
    DynTensor, QuantType, QuantizedStorage,
};
use crate::var_builder::{
    verify_mapper_coverage, HfToNnMapper, TensorBackend, TensorMapBackend, VarBuilder,
    WeightNameMapper,
};
use crate::{DType, Device, TensorError};

// ===========================================================================
// A. Multi-file safetensors loading (sharded checkpoints)
// ===========================================================================

#[test]
fn test_multi_file_safetensors_loading_merge_disjoint() {
    // Simulate sharded checkpoint: two safetensors files with disjoint keys.
    let mut shard1 = HashMap::new();
    shard1.insert(
        "encoder.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    shard1.insert(
        "encoder.bias".to_string(),
        DynTensor::new(&[0.1, 0.2], &[2], &Device::Cpu).unwrap(),
    );

    let mut shard2 = HashMap::new();
    shard2.insert(
        "decoder.weight".to_string(),
        DynTensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    shard2.insert(
        "decoder.bias".to_string(),
        DynTensor::new(&[0.5, 0.6], &[2], &Device::Cpu).unwrap(),
    );

    let bytes1 = tensors_to_safetensors_bytes(&shard1).unwrap();
    let bytes2 = tensors_to_safetensors_bytes(&shard2).unwrap();

    // Merge loaded shards into a single map.
    let mut merged = load_safetensors_from_bytes(&bytes1).unwrap();
    let shard2_loaded = load_safetensors_from_bytes(&bytes2).unwrap();
    merged.extend(shard2_loaded);

    assert_eq!(merged.len(), 4);
    let vb = VarBuilder::from_tensors(merged, DType::F32, &Device::Cpu);
    let enc_w = vb.pp("encoder").get(&[2, 2], "weight").unwrap();
    let dec_w = vb.pp("decoder").get(&[2, 2], "weight").unwrap();
    assert_eq!(
        enc_w.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        dec_w.to_flat_vec::<f32>().unwrap(),
        vec![5.0, 6.0, 7.0, 8.0]
    );
}

#[test]
fn test_multi_file_safetensors_loading_three_shards() {
    // Three-shard pattern (common for large models like Llama-70B).
    let shards: Vec<HashMap<String, DynTensor>> = (0..3)
        .map(|i| {
            let mut shard = HashMap::new();
            let data: Vec<f32> = (0..4).map(|j| (i * 4 + j) as f32).collect();
            shard.insert(
                format!("layers.{i}.weight"),
                DynTensor::new(&data, &[2, 2], &Device::Cpu).unwrap(),
            );
            shard
        })
        .collect();

    let mut merged = HashMap::new();
    for shard in &shards {
        let bytes = tensors_to_safetensors_bytes(shard).unwrap();
        let loaded = load_safetensors_from_bytes(&bytes).unwrap();
        merged.extend(loaded);
    }

    assert_eq!(merged.len(), 3);
    let vb = VarBuilder::from_tensors(merged, DType::F32, &Device::Cpu);
    for i in 0..3 {
        let t = vb
            .pp("layers")
            .pp(i.to_string())
            .get(&[2, 2], "weight")
            .unwrap();
        let expected: Vec<f32> = (0..4).map(|j| (i * 4 + j) as f32).collect();
        assert_eq!(t.to_flat_vec::<f32>().unwrap(), expected);
    }
}

#[test]
fn test_multi_file_safetensors_file_roundtrip() {
    // Write two files, load both, merge into VarBuilder.
    let dir = std::env::temp_dir().join("nn_test_multi_shard");
    std::fs::create_dir_all(&dir).unwrap();
    let path1 = dir.join("shard_00001.safetensors");
    let path2 = dir.join("shard_00002.safetensors");

    let mut shard1 = HashMap::new();
    shard1.insert(
        "model.embed.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap(),
    );
    let mut shard2 = HashMap::new();
    shard2.insert(
        "model.head.weight".to_string(),
        DynTensor::new(&[4.0, 5.0, 6.0], &[3], &Device::Cpu).unwrap(),
    );

    save_safetensors(&shard1, &path1).unwrap();
    save_safetensors(&shard2, &path2).unwrap();

    let mut merged = load_safetensors(&path1).unwrap();
    merged.extend(load_safetensors(&path2).unwrap());

    let vb = VarBuilder::from_tensors(merged, DType::F32, &Device::Cpu);
    let embed = vb.pp("model").pp("embed").get(&[3], "weight").unwrap();
    let head = vb.pp("model").pp("head").get(&[3], "weight").unwrap();
    assert_eq!(embed.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(head.to_flat_vec::<f32>().unwrap(), vec![4.0, 5.0, 6.0]);

    let _ = std::fs::remove_dir_all(&dir);
}

// ===========================================================================
// B. Shape validation for real model configurations
// ===========================================================================

#[test]
fn test_transformer_config_shapes_small() {
    // Small transformer: hidden=64, heads=4, head_dim=16, ffn=256, vocab=1000
    let hidden = 64;
    let heads = 4;
    let head_dim = hidden / heads;
    let ffn = 256;
    let vocab = 1000;
    let num_layers = 2;

    let mut tensors = HashMap::new();
    // Embedding
    tensors.insert(
        "embed_tokens.weight".to_string(),
        DynTensor::zeros(&[vocab, hidden], DType::F32, &Device::Cpu).unwrap(),
    );
    // Per-layer weights
    for i in 0..num_layers {
        let prefix = format!("layers.{i}");
        // Self-attention Q, K, V, O projections
        for proj in &["q_proj", "k_proj", "v_proj"] {
            tensors.insert(
                format!("{prefix}.self_attn.{proj}.weight"),
                DynTensor::zeros(&[heads * head_dim, hidden], DType::F32, &Device::Cpu).unwrap(),
            );
        }
        tensors.insert(
            format!("{prefix}.self_attn.o_proj.weight"),
            DynTensor::zeros(&[hidden, heads * head_dim], DType::F32, &Device::Cpu).unwrap(),
        );
        // MLP gate/up/down
        tensors.insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            DynTensor::zeros(&[ffn, hidden], DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.mlp.up_proj.weight"),
            DynTensor::zeros(&[ffn, hidden], DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            DynTensor::zeros(&[hidden, ffn], DType::F32, &Device::Cpu).unwrap(),
        );
        // Layer norms
        tensors.insert(
            format!("{prefix}.input_layernorm.weight"),
            DynTensor::ones(&[hidden], DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            DynTensor::ones(&[hidden], DType::F32, &Device::Cpu).unwrap(),
        );
    }
    // LM head
    tensors.insert(
        "lm_head.weight".to_string(),
        DynTensor::zeros(&[vocab, hidden], DType::F32, &Device::Cpu).unwrap(),
    );

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    // Validate all shapes match expected config.
    let embed = vb.get(&[vocab, hidden], "embed_tokens.weight").unwrap();
    assert_eq!(embed.dims(), &[vocab, hidden]);

    for i in 0..num_layers {
        let layer = vb.pp("layers").pp(i.to_string());
        let attn = layer.pp("self_attn");
        let q = attn.get(&[hidden, hidden], "q_proj.weight").unwrap();
        assert_eq!(q.dims(), &[hidden, hidden]);

        let mlp = layer.pp("mlp");
        let gate = mlp.get(&[ffn, hidden], "gate_proj.weight").unwrap();
        assert_eq!(gate.dims(), &[ffn, hidden]);
        let down = mlp.get(&[hidden, ffn], "down_proj.weight").unwrap();
        assert_eq!(down.dims(), &[hidden, ffn]);

        let ln = layer.get(&[hidden], "input_layernorm.weight").unwrap();
        assert_eq!(ln.dims(), &[hidden]);
    }
}

#[test]
fn test_conv_model_shapes() {
    // Conv-based model: out_ch=16, in_ch=3, kernel=3x3, with batch norm
    let in_ch = 3;
    let out_ch = 16;
    let kernel = 3;

    let mut tensors = HashMap::new();
    tensors.insert(
        "conv.weight".to_string(),
        DynTensor::zeros(&[out_ch, in_ch, kernel, kernel], DType::F32, &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "conv.bias".to_string(),
        DynTensor::zeros(&[out_ch], DType::F32, &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "bn.weight".to_string(),
        DynTensor::ones(&[out_ch], DType::F32, &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "bn.bias".to_string(),
        DynTensor::zeros(&[out_ch], DType::F32, &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "bn.running_mean".to_string(),
        DynTensor::zeros(&[out_ch], DType::F32, &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "bn.running_var".to_string(),
        DynTensor::ones(&[out_ch], DType::F32, &Device::Cpu).unwrap(),
    );

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let conv_w = vb
        .pp("conv")
        .get(&[out_ch, in_ch, kernel, kernel], "weight")
        .unwrap();
    assert_eq!(conv_w.dims(), &[16, 3, 3, 3]);
    let bn_w = vb.pp("bn").get(&[out_ch], "weight").unwrap();
    assert_eq!(bn_w.dims(), &[16]);
    let bn_rm = vb.pp("bn").get(&[out_ch], "running_mean").unwrap();
    assert_eq!(bn_rm.dims(), &[16]);
}

#[test]
fn test_shape_mismatch_on_wrong_config() {
    // Model code expects [512, 256] but checkpoint has [256, 512] (transposed).
    let mut tensors = HashMap::new();
    tensors.insert(
        "proj.weight".to_string(),
        DynTensor::zeros(&[256, 512], DType::F32, &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let err = vb.pp("proj").get(&[512, 256], "weight").unwrap_err();
    match err {
        TensorError::ShapeMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, vec![512, 256]);
            assert_eq!(actual, vec![256, 512]);
        }
        other => panic!("expected ShapeMismatch, got: {other:?}"),
    }
}

// ===========================================================================
// C. Missing weight detection and graceful error reporting
// ===========================================================================

#[test]
fn test_missing_required_weight_reports_full_path() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "encoder.layer.0.weight".to_string(),
        DynTensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    // Bias is missing.
    let err = vb
        .pp("encoder")
        .pp("layer")
        .pp("0")
        .get(&[4], "bias")
        .unwrap_err();
    match err {
        TensorError::TensorNotFound { name } => {
            assert_eq!(name, "encoder.layer.0.bias");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

#[test]
fn test_missing_weight_with_name_mapping_reports_mapped_key() {
    let tensors = HashMap::new();
    let mapper = HfToNnMapper::decoder_transformer();
    let vb =
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu).with_weight_name_mapper(mapper);

    let err = vb
        .pp("layers")
        .pp("0")
        .pp("attn")
        .get(&[64, 64], "q.weight")
        .unwrap_err();

    // The error should contain the mapped (HF) key, not the NN key.
    match err {
        TensorError::TensorNotFound { name } => {
            assert_eq!(name, "model.layers.0.self_attn.q_proj.weight");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

#[test]
fn test_collect_all_missing_weights() {
    // Pattern: before loading, check all required weights exist.
    let mut tensors = HashMap::new();
    tensors.insert(
        "layer.0.weight".to_string(),
        DynTensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    let required = ["layer.0.weight",
        "layer.0.bias",
        "layer.1.weight",
        "layer.1.bias"];

    let missing: Vec<&str> = required
        .iter()
        .filter(|name| !vb.contains_tensor(name))
        .copied()
        .collect();

    assert_eq!(
        missing,
        vec!["layer.0.bias", "layer.1.weight", "layer.1.bias"]
    );
}

#[test]
fn test_optional_bias_graceful_handling() {
    // Pattern: model tries to load bias, falls back if not present.
    let mut tensors = HashMap::new();
    tensors.insert(
        "proj.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let proj = vb.pp("proj");

    // Weight loads fine.
    let w = proj.get(&[2, 2], "weight").unwrap();
    assert_eq!(w.dims(), &[2, 2]);

    // Bias is optional: check first, skip if absent.
    let has_bias = proj.contains_tensor("bias");
    assert!(!has_bias);

    // Or attempt load and handle error.
    let bias_result = proj.get(&[2], "bias");
    assert!(bias_result.is_err());
}

// ===========================================================================
// D. Extra/unused weight detection patterns
// ===========================================================================

#[test]
fn test_detect_unused_weights_via_tensor_names() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "encoder.weight".to_string(),
        DynTensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "encoder.bias".to_string(),
        DynTensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "extra_unused.weight".to_string(),
        DynTensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap(),
    );

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    // Model only uses encoder.weight and encoder.bias.
    let model_keys: std::collections::HashSet<&str> =
        ["encoder.weight", "encoder.bias"].into_iter().collect();

    let all_checkpoint_keys: Vec<String> = vb.tensor_names();
    let unused: Vec<&String> = all_checkpoint_keys
        .iter()
        .filter(|k| !model_keys.contains(k.as_str()))
        .collect();

    assert_eq!(unused.len(), 1);
    assert_eq!(unused[0], "extra_unused.weight");
}

#[test]
fn test_detect_unused_weights_with_mapper() {
    // When using a mapper, we can check coverage: which checkpoint keys have
    // no corresponding NN model key.
    let mapper = HfToNnMapper::new().with_prefix_rule("model", "m");

    let checkpoint_names = vec![
        "model.weight".to_string(),
        "model.bias".to_string(),
        "model.unused_param".to_string(),
    ];
    let nn_names = vec!["m.weight".to_string(), "m.bias".to_string()];

    // verify_mapper_coverage checks NN -> checkpoint direction.
    let missing = verify_mapper_coverage(&nn_names, &checkpoint_names, &mapper);
    assert!(missing.is_empty(), "all NN names should resolve");

    // For "extra" detection: check checkpoint keys not covered by any NN key.
    let mapped_checkpoint_keys: std::collections::HashSet<String> =
        nn_names.iter().map(|n| mapper.map_name(n)).collect();

    let extra: Vec<&String> = checkpoint_names
        .iter()
        .filter(|k| !mapped_checkpoint_keys.contains(k.as_str()))
        .collect();

    assert_eq!(extra.len(), 1);
    assert_eq!(extra[0], "model.unused_param");
}

#[test]
fn test_no_unused_weights_reports_empty() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "w".to_string(),
        DynTensor::zeros(&[2], DType::F32, &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "b".to_string(),
        DynTensor::zeros(&[2], DType::F32, &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    let model_keys: std::collections::HashSet<&str> = ["w", "b"].into_iter().collect();
    let all_keys = vb.tensor_names();
    let unused: Vec<&String> = all_keys
        .iter()
        .filter(|k| !model_keys.contains(k.as_str()))
        .collect();

    assert!(unused.is_empty());
}

// ===========================================================================
// E. Quantized weight loading (Q4_0, Q8_0 block quantization)
// ===========================================================================

#[test]
fn test_quantized_storage_q8_0_basic_roundtrip() {
    // Q8_0: 34 bytes per block of 32 elements.
    // Layout: [2 bytes f16 scale][32 bytes i8 values]
    let num_elements = 32;
    let scale: f32 = 0.5;
    let scale_f16 = half::f16::from_f32(scale);

    let mut block_data = Vec::new();
    block_data.extend_from_slice(&scale_f16.to_le_bytes());
    // 32 signed int8 values: 0, 1, 2, ..., 31
    for i in 0..32u8 {
        block_data.push(i);
    }
    assert_eq!(block_data.len(), 34);

    let qs = QuantizedStorage::new(block_data, &[num_elements], QuantType::Q8_0).unwrap();
    assert_eq!(qs.shape(), &[num_elements]);
    assert_eq!(qs.quant_type(), QuantType::Q8_0);

    let deq = qs.dequantize().unwrap();
    assert_eq!(deq.len(), num_elements);

    // Q8_0 dequant: val = scale * q_int8
    let scale_actual = scale_f16.to_f32();
    for i in 0..32 {
        let expected = scale_actual * (i as f32);
        let got = deq[[i]];
        assert!(
            (got - expected).abs() < 1e-3,
            "Q8_0 deq[{i}]: got {got}, expected {expected}"
        );
    }
}

#[test]
fn test_quantized_storage_q4_0_basic() {
    // Q4_0: 18 bytes per block of 32 elements.
    // Layout: [2 bytes f16 scale][16 bytes: 32x4bit packed 2/byte]
    let num_elements = 32;
    let scale: f32 = 1.0;
    let scale_f16 = half::f16::from_f32(scale);

    let mut block_data = Vec::new();
    block_data.extend_from_slice(&scale_f16.to_le_bytes());
    // 16 bytes of nibbles. Each byte = 2 values.
    // For simplicity: all nibbles = 8 => dequant val = scale * (8-8) = 0.
    // 16 bytes, each 0x88 (lo=8, hi=8).
    block_data.extend(std::iter::repeat_n(0x88u8, 16));
    assert_eq!(block_data.len(), 18);

    let qs = QuantizedStorage::new(block_data, &[num_elements], QuantType::Q4_0).unwrap();
    let deq = qs.dequantize().unwrap();
    assert_eq!(deq.len(), num_elements);
    // All values should be 0 (scale * (8-8) = 0).
    for i in 0..num_elements {
        assert!(
            deq[[i]].abs() < 1e-6,
            "Q4_0 deq[{i}] should be 0, got {}",
            deq[[i]]
        );
    }
}

#[test]
fn test_quantized_storage_invalid_data_length() {
    // Q8_0 expects 34 bytes for 32 elements; provide wrong length.
    let data = vec![0u8; 30]; // too short
    let err = QuantizedStorage::new(data, &[32], QuantType::Q8_0).unwrap_err();
    match err {
        TensorError::DataLengthMismatch { expected, actual } => {
            assert_eq!(expected, 34);
            assert_eq!(actual, 30);
        }
        other => panic!("expected DataLengthMismatch, got: {other:?}"),
    }
}

#[test]
fn test_quantized_storage_non_block_aligned_elements() {
    // 33 elements not a multiple of block_size=32.
    let data = vec![0u8; 100];
    let err = QuantizedStorage::new(data, &[33], QuantType::Q8_0).unwrap_err();
    assert!(
        format!("{err}").contains("block size"),
        "error should mention block size: {err}"
    );
}

#[test]
fn test_quantized_storage_2d_shape() {
    // Q8_0 with 2D shape [2, 32] = 64 elements = 2 blocks = 68 bytes.
    let scale_f16 = half::f16::from_f32(1.0);
    let mut data = Vec::new();
    for _ in 0..2 {
        data.extend_from_slice(&scale_f16.to_le_bytes());
        data.extend(vec![0u8; 32]);
    }
    assert_eq!(data.len(), 68);

    let qs = QuantizedStorage::new(data, &[2, 32], QuantType::Q8_0).unwrap();
    assert_eq!(qs.shape(), &[2, 32]);
    let deq = qs.dequantize().unwrap();
    assert_eq!(deq.shape(), &[2, 32]);
}

#[test]
fn test_quant_type_expected_bytes() {
    assert_eq!(QuantType::Q4_0.expected_bytes(32), Some(18));
    assert_eq!(QuantType::Q4_0.expected_bytes(64), Some(36));
    assert_eq!(QuantType::Q4_0.expected_bytes(0), Some(0));
    assert_eq!(QuantType::Q4_0.expected_bytes(33), None); // not block-aligned

    assert_eq!(QuantType::Q8_0.expected_bytes(32), Some(34));
    assert_eq!(QuantType::Q8_0.expected_bytes(64), Some(68));
    assert_eq!(QuantType::Q8_0.expected_bytes(0), Some(0));
    assert_eq!(QuantType::Q8_0.expected_bytes(31), None);

    assert_eq!(QuantType::Q4_1.expected_bytes(32), Some(20));
    assert_eq!(QuantType::Q4_1.expected_bytes(64), Some(40));
}

#[test]
fn test_quant_type_block_properties() {
    assert_eq!(QuantType::Q4_0.block_size(), 32);
    assert_eq!(QuantType::Q4_0.block_bytes(), 18);
    assert_eq!(QuantType::Q4_1.block_size(), 32);
    assert_eq!(QuantType::Q4_1.block_bytes(), 20);
    assert_eq!(QuantType::Q8_0.block_size(), 32);
    assert_eq!(QuantType::Q8_0.block_bytes(), 34);
}

// ===========================================================================
// F. Memory-mapped loading simulation (file-based load pattern)
// ===========================================================================

#[test]
fn test_mmap_style_file_load_preserves_data() {
    // Simulates mmap-based loading: write safetensors, load (which reads from
    // file just like mmap would), verify byte-exact.
    let dir = std::env::temp_dir().join("nn_test_mmap_sim");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("weights.safetensors");

    let original_data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.001).collect();
    let mut tensors = HashMap::new();
    tensors.insert(
        "model.weight".to_string(),
        DynTensor::new(&original_data, &[16, 16], &Device::Cpu).unwrap(),
    );
    save_safetensors(&tensors, &path).unwrap();

    let loaded = load_safetensors(&path).unwrap();
    let t = &loaded["model.weight"];
    assert_eq!(t.dims(), &[16, 16]);
    let loaded_data = t.to_flat_vec::<f32>().unwrap();
    for (i, (&got, &expected)) in loaded_data.iter().zip(original_data.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-7,
            "mmap load mismatch at [{i}]: got {got}, expected {expected}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_mmap_style_load_large_tensor() {
    // Larger tensor to exercise multi-page memory regions.
    let dir = std::env::temp_dir().join("nn_test_mmap_large");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("large_weights.safetensors");

    let size = 4096;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.0001).collect();
    let mut tensors = HashMap::new();
    tensors.insert(
        "large.weight".to_string(),
        DynTensor::new(&data, &[64, 64], &Device::Cpu).unwrap(),
    );
    save_safetensors(&tensors, &path).unwrap();

    // Verify file size is reasonable (header + 4096*4 bytes data).
    let file_size = std::fs::metadata(&path).unwrap().len();
    assert!(
        file_size >= 4096 * 4,
        "file should be at least 16KB, got {file_size}"
    );

    let loaded = load_safetensors(&path).unwrap();
    let t = &loaded["large.weight"];
    assert_eq!(t.dims(), &[64, 64]);
    let loaded_data = t.to_flat_vec::<f32>().unwrap();
    assert_eq!(loaded_data.len(), size);
    // Spot check first and last.
    assert!((loaded_data[0] - 0.0).abs() < 1e-7);
    assert!((loaded_data[size - 1] - ((size - 1) as f32 * 0.0001)).abs() < 1e-4);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_mmap_style_load_multiple_tensors() {
    // Multiple tensors in one file, simulating a real model checkpoint.
    let dir = std::env::temp_dir().join("nn_test_mmap_multi");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");

    let mut tensors = HashMap::new();
    for i in 0..5 {
        tensors.insert(
            format!("layer.{i}.weight"),
            DynTensor::new(&[(i as f32) + 0.1, (i as f32) + 0.2], &[2], &Device::Cpu).unwrap(),
        );
    }
    save_safetensors(&tensors, &path).unwrap();

    let loaded = load_safetensors(&path).unwrap();
    assert_eq!(loaded.len(), 5);

    let vb = VarBuilder::from_tensors(loaded, DType::F32, &Device::Cpu);
    for i in 0..5 {
        let t = vb
            .pp("layer")
            .pp(i.to_string())
            .get(&[2], "weight")
            .unwrap();
        let expected = [(i as f32) + 0.1, (i as f32) + 0.2];
        let got = t.to_flat_vec::<f32>().unwrap();
        for (j, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "layer.{i}.weight[{j}]: got {g}, expected {e}"
            );
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
}

// ===========================================================================
// G. PyTorch-to-nn naming conventions for real model families
// ===========================================================================

#[test]
fn test_qwen3_naming_identity() {
    // Qwen3 NN model matches HF naming, so mapper is identity.
    let mapper = HfToNnMapper::qwen3();

    let hf_names = vec![
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ];

    for name in &hf_names {
        assert_eq!(
            mapper.map_name(name),
            *name,
            "Qwen3 mapper should be identity for {name}"
        );
    }
}

#[test]
fn test_siglip2_granite_docling_prefix_strip() {
    let mapper = HfToNnMapper::siglip2_granite_docling();

    // NN names (without "model.vision_model." prefix)
    // should map to HF names (with prefix).
    assert_eq!(
        mapper.map_name("encoder.layers.0.self_attn.q_proj.weight"),
        "model.vision_model.encoder.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        mapper.map_name("embeddings.patch_embedding.weight"),
        "model.vision_model.embeddings.patch_embedding.weight"
    );
    assert_eq!(
        mapper.map_name("post_layernorm.weight"),
        "model.vision_model.post_layernorm.weight"
    );
}

#[test]
fn test_decoder_transformer_full_mapping() {
    let mapper = HfToNnMapper::decoder_transformer();

    // NN names -> HF checkpoint names.
    assert_eq!(
        mapper.map_name("layers.0.attn.q.weight"),
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.attn.k.weight"),
        "model.layers.0.self_attn.k_proj.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.attn.v.weight"),
        "model.layers.0.self_attn.v_proj.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.attn.o.weight"),
        "model.layers.0.self_attn.o_proj.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.mlp.gate.weight"),
        "model.layers.0.mlp.gate_proj.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.mlp.up.weight"),
        "model.layers.0.mlp.up_proj.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.mlp.down.weight"),
        "model.layers.0.mlp.down_proj.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.ln1.weight"),
        "model.layers.0.input_layernorm.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.ln2.weight"),
        "model.layers.0.post_attention_layernorm.weight"
    );
}

#[test]
fn test_decoder_transformer_with_varbuilder_integration() {
    // End-to-end: HF checkpoint names in backend, NN model code uses short names.
    let mut hf_tensors = HashMap::new();
    hf_tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    hf_tensors.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        DynTensor::new(&[3.0, 4.0], &[2], &Device::Cpu).unwrap(),
    );
    hf_tensors.insert(
        "model.layers.0.input_layernorm.weight".to_string(),
        DynTensor::new(&[5.0, 6.0], &[2], &Device::Cpu).unwrap(),
    );

    let mapper = HfToNnMapper::decoder_transformer();
    let vb = VarBuilder::from_tensors(hf_tensors, DType::F32, &Device::Cpu)
        .with_weight_name_mapper(mapper);

    let layer = vb.pp("layers").pp("0");
    let q = layer.pp("attn").get(&[2], "q.weight").unwrap();
    assert_eq!(q.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);

    let gate = layer.pp("mlp").get(&[2], "gate.weight").unwrap();
    assert_eq!(gate.to_flat_vec::<f32>().unwrap(), vec![3.0, 4.0]);

    let ln = layer.get(&[2], "ln1.weight").unwrap();
    assert_eq!(ln.to_flat_vec::<f32>().unwrap(), vec![5.0, 6.0]);
}

#[test]
fn test_mapper_coverage_for_multi_layer_model() {
    let mapper = HfToNnMapper::decoder_transformer();
    let num_layers = 4;

    let mut checkpoint_names = Vec::new();
    let mut nn_names = Vec::new();

    for i in 0..num_layers {
        for suffix in &[
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ] {
            checkpoint_names.push(format!("model.layers.{i}.{suffix}"));
        }
        for suffix in &[
            "attn.q.weight",
            "attn.k.weight",
            "attn.v.weight",
            "attn.o.weight",
            "mlp.gate.weight",
            "mlp.up.weight",
            "mlp.down.weight",
            "ln1.weight",
            "ln2.weight",
        ] {
            nn_names.push(format!("layers.{i}.{suffix}"));
        }
    }

    let missing = verify_mapper_coverage(&nn_names, &checkpoint_names, &mapper);
    assert!(
        missing.is_empty(),
        "all {num_layers}-layer decoder weights should be covered: {missing:?}"
    );
}

// ===========================================================================
// H. Weight dtype conversion during import (F32 <-> BF16/F16)
// ===========================================================================

#[test]
fn test_dtype_conversion_f32_to_bf16_via_varbuilder() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    // VarBuilder requests BF16 dtype.
    let vb = VarBuilder::from_tensors(tensors, DType::BF16, &Device::Cpu);
    let t = vb.get(&[2, 2], "w").unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    assert_eq!(t.dims(), &[2, 2]);

    // Convert back to f32 for value check.
    let f32_vals = t.to_f32_array().unwrap();
    let vals: Vec<f32> = f32_vals.iter().copied().collect();
    assert!((vals[0] - 1.0).abs() < 0.01);
    assert!((vals[3] - 4.0).abs() < 0.01);
}

#[test]
fn test_dtype_conversion_f32_to_f16_via_varbuilder() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "w".to_string(),
        DynTensor::new(&[0.5, -1.5, 2.25, 3.75], &[4], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F16, &Device::Cpu);
    let t = vb.get(&[4], "w").unwrap();
    assert_eq!(t.dtype(), DType::F16);

    let f32_vals = t.to_f32_array().unwrap();
    let vals: Vec<f32> = f32_vals.iter().copied().collect();
    assert!((vals[0] - 0.5).abs() < 0.01);
    assert!((vals[1] - (-1.5)).abs() < 0.01);
    assert!((vals[2] - 2.25).abs() < 0.01);
    assert!((vals[3] - 3.75).abs() < 0.01);
}

#[test]
fn test_dtype_conversion_bf16_safetensors_roundtrip() {
    // Save as F32, load into BF16 VarBuilder.
    let mut tensors = HashMap::new();
    tensors.insert(
        "param".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();

    let vb = VarBuilder::from_tensors(loaded, DType::BF16, &Device::Cpu);
    let t = vb.get(&[2], "param").unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    let f32_vals = t.to_f32_array().unwrap();
    let vals: Vec<f32> = f32_vals.iter().copied().collect();
    assert!((vals[0] - 1.0).abs() < 0.01);
    assert!((vals[1] - 2.0).abs() < 0.01);
}

#[test]
fn test_to_dtype_changes_subsequent_loads() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    // F32 load.
    let t_f32 = vb.get(&[2], "w").unwrap();
    assert_eq!(t_f32.dtype(), DType::F32);

    // Switch to BF16.
    let vb_bf16 = vb.to_dtype(DType::BF16);
    let t_bf16 = vb_bf16.get(&[2], "w").unwrap();
    assert_eq!(t_bf16.dtype(), DType::BF16);

    // Original VarBuilder unchanged.
    assert_eq!(vb.dtype(), DType::F32);
}

#[test]
fn test_mixed_precision_policy_effective_dtype() {
    use crate::mixed_precision::MixedPrecisionPolicy;

    let mut tensors = HashMap::new();
    tensors.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let vb =
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu).with_precision_policy(policy);

    // effective_weight_dtype should return BF16 (from policy).
    assert_eq!(vb.effective_weight_dtype(), DType::BF16);

    // But the VarBuilder's dtype is still F32 (policy affects weight loading,
    // not the base dtype).
    assert_eq!(vb.dtype(), DType::F32);

    // Child inherits policy.
    let child = vb.pp("encoder");
    assert_eq!(child.effective_weight_dtype(), DType::BF16);
}

// ===========================================================================
// I. Multi-layer transformer weight pattern loading
// ===========================================================================

#[test]
fn test_multi_layer_transformer_all_weights_accessible() {
    let num_layers = 6;
    let hidden = 32;
    let ffn = 128;

    let mut tensors = HashMap::new();
    for i in 0..num_layers {
        let prefix = format!("model.layers.{i}");
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            tensors.insert(
                format!("{prefix}.self_attn.{proj}.weight"),
                DynTensor::zeros(&[hidden, hidden], DType::F32, &Device::Cpu).unwrap(),
            );
        }
        tensors.insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            DynTensor::zeros(&[ffn, hidden], DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.mlp.up_proj.weight"),
            DynTensor::zeros(&[ffn, hidden], DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            DynTensor::zeros(&[hidden, ffn], DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.input_layernorm.weight"),
            DynTensor::ones(&[hidden], DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            DynTensor::ones(&[hidden], DType::F32, &Device::Cpu).unwrap(),
        );
    }
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        DynTensor::zeros(&[1000, hidden], DType::F32, &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "model.norm.weight".to_string(),
        DynTensor::ones(&[hidden], DType::F32, &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "lm_head.weight".to_string(),
        DynTensor::zeros(&[1000, hidden], DType::F32, &Device::Cpu).unwrap(),
    );

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    // Access every weight via hierarchical pp().
    for i in 0..num_layers {
        let layer = vb.pp("model").pp("layers").pp(i.to_string());
        let attn = layer.pp("self_attn");
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let w = attn
                .get(&[hidden, hidden], &format!("{proj}.weight"))
                .unwrap();
            assert_eq!(w.dims(), &[hidden, hidden], "layer {i} {proj}");
        }
        let mlp = layer.pp("mlp");
        let gate = mlp.get(&[ffn, hidden], "gate_proj.weight").unwrap();
        assert_eq!(gate.dims(), &[ffn, hidden]);
        let down = mlp.get(&[hidden, ffn], "down_proj.weight").unwrap();
        assert_eq!(down.dims(), &[hidden, ffn]);
    }

    let embed = vb
        .pp("model")
        .get(&[1000, hidden], "embed_tokens.weight")
        .unwrap();
    assert_eq!(embed.dims(), &[1000, hidden]);
    let final_norm = vb.pp("model").get(&[hidden], "norm.weight").unwrap();
    assert_eq!(final_norm.dims(), &[hidden]);
    let lm_head = vb.get(&[1000, hidden], "lm_head.weight").unwrap();
    assert_eq!(lm_head.dims(), &[1000, hidden]);
}

#[test]
fn test_multi_layer_with_decoder_transformer_mapper() {
    let num_layers = 3;
    let hidden = 16;

    // Build HF-style checkpoint.
    let mut hf_tensors = HashMap::new();
    for i in 0..num_layers {
        hf_tensors.insert(
            format!("model.layers.{i}.self_attn.q_proj.weight"),
            DynTensor::full(&[hidden, hidden], f64::from(i + 1), DType::F32, &Device::Cpu).unwrap(),
        );
        hf_tensors.insert(
            format!("model.layers.{i}.input_layernorm.weight"),
            DynTensor::ones(&[hidden], DType::F32, &Device::Cpu).unwrap(),
        );
    }

    let mapper = HfToNnMapper::decoder_transformer();
    let vb = VarBuilder::from_tensors(hf_tensors, DType::F32, &Device::Cpu)
        .with_weight_name_mapper(mapper);

    // Access via NN naming convention.
    for i in 0..num_layers {
        let layer = vb.pp("layers").pp(i.to_string());
        let q = layer.pp("attn").get(&[hidden, hidden], "q.weight").unwrap();
        let expected_val = (i + 1) as f32;
        let data = q.to_flat_vec::<f32>().unwrap();
        assert!(
            (data[0] - expected_val).abs() < 1e-6,
            "layer {i}: expected {expected_val}, got {}",
            data[0]
        );

        let ln = layer.get(&[hidden], "ln1.weight").unwrap();
        let ln_data = ln.to_flat_vec::<f32>().unwrap();
        assert!(
            (ln_data[0] - 1.0).abs() < 1e-6,
            "layer {i} layernorm should be 1.0"
        );
    }
}

// ===========================================================================
// J. Edge cases for import infrastructure
// ===========================================================================

#[test]
fn test_empty_checkpoint_reports_all_missing() {
    let tensors = HashMap::new();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    let required_weights = vec![
        "embed.weight",
        "layer.0.weight",
        "layer.0.bias",
        "head.weight",
    ];

    for name in &required_weights {
        assert!(
            !vb.contains_tensor(name),
            "{name} should not exist in empty backend"
        );
    }

    let missing: Vec<&&str> = required_weights
        .iter()
        .filter(|name| !vb.contains_tensor(name))
        .collect();
    assert_eq!(missing.len(), 4);
}

#[test]
fn test_special_characters_in_weight_names() {
    // Some checkpoints have unusual characters in names.
    let mut tensors = HashMap::new();
    tensors.insert(
        "model/encoder/layer_0/weight:0".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    // Direct access without pp (since the key contains "/" and ":").
    let t = vb.get(&[2], "model/encoder/layer_0/weight:0").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn test_unicode_weight_names() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "layer_alpha.weight".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let t = vb.pp("layer_alpha").get(&[1], "weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0]);
}

#[test]
fn test_very_long_weight_name() {
    let long_prefix = "model.encoder.transformer.blocks.layer_99.sublayer.attention.multi_head";
    let full_name = format!("{long_prefix}.q_proj.weight");
    let mut tensors = HashMap::new();
    tensors.insert(
        full_name.clone(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let t = vb.get(&[1], &full_name).unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0]);
}

#[test]
fn test_duplicate_key_last_shard_wins() {
    // When merging shards with duplicate keys, HashMap::extend means last wins.
    let mut shard1 = HashMap::new();
    shard1.insert(
        "shared.weight".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let mut shard2 = HashMap::new();
    shard2.insert(
        "shared.weight".to_string(),
        DynTensor::new(&[2.0], &[1], &Device::Cpu).unwrap(),
    );

    let bytes1 = tensors_to_safetensors_bytes(&shard1).unwrap();
    let bytes2 = tensors_to_safetensors_bytes(&shard2).unwrap();

    let mut merged = load_safetensors_from_bytes(&bytes1).unwrap();
    merged.extend(load_safetensors_from_bytes(&bytes2).unwrap());

    let vb = VarBuilder::from_tensors(merged, DType::F32, &Device::Cpu);
    let t = vb.get(&[1], "shared.weight").unwrap();
    // shard2 was extended last, so its value should win.
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![2.0]);
}

// ===========================================================================
// K. Custom backend for tracking loaded weights
// ===========================================================================

#[test]
fn test_tracking_backend_records_accessed_keys() {
    use std::sync::Mutex;

    /// Backend that records which keys are accessed.
    struct TrackingBackend {
        inner: TensorMapBackend,
        accessed: Mutex<Vec<String>>,
    }

    impl TensorBackend for TrackingBackend {
        fn get(
            &self,
            dims: &[usize],
            name: &str,
            dtype: DType,
            device: &Device,
        ) -> crate::Result<DynTensor> {
            self.accessed.lock().unwrap().push(name.to_string());
            self.inner.get(dims, name, dtype, device)
        }

        fn get_unchecked(
            &self,
            name: &str,
            dtype: DType,
            device: &Device,
        ) -> crate::Result<DynTensor> {
            self.accessed.lock().unwrap().push(name.to_string());
            self.inner.get_unchecked(name, dtype, device)
        }

        fn contains_tensor(&self, name: &str) -> bool {
            self.inner.contains_tensor(name)
        }

        fn tensor_names(&self) -> Vec<String> {
            self.inner.tensor_names()
        }
    }

    let mut map = HashMap::new();
    map.insert(
        "encoder.weight".to_string(),
        DynTensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap(),
    );
    map.insert(
        "encoder.bias".to_string(),
        DynTensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap(),
    );
    map.insert(
        "unused.weight".to_string(),
        DynTensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap(),
    );

    let backend = Arc::new(TrackingBackend {
        inner: TensorMapBackend::new(map),
        accessed: Mutex::new(Vec::new()),
    });

    let vb = VarBuilder::from_backend(
        Arc::clone(&backend) as Arc<dyn TensorBackend>,
        DType::F32,
        Device::Cpu,
    );

    // Model loads encoder weights.
    let _ = vb.pp("encoder").get(&[4], "weight").unwrap();
    let _ = vb.pp("encoder").get(&[4], "bias").unwrap();

    let accessed = backend.accessed.lock().unwrap();
    assert_eq!(accessed.len(), 2);
    assert_eq!(accessed[0], "encoder.weight");
    assert_eq!(accessed[1], "encoder.bias");

    // "unused.weight" was never accessed.
    let all_names = backend.tensor_names();
    let unused: Vec<&String> = all_names.iter().filter(|k| !accessed.contains(k)).collect();
    assert_eq!(unused.len(), 1);
    assert_eq!(unused[0], "unused.weight");
}

// ===========================================================================
// L. Safetensors BF16/F16 import through VarBuilder
// ===========================================================================

#[test]
fn test_bf16_safetensors_into_varbuilder() {
    // Build BF16 safetensors bytes, load, put into VarBuilder.
    let values = [1.0f32, -2.5, 3.75, 0.0];
    let bf16_bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
        .collect();

    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![4], &bf16_bytes)
        .unwrap();
    let bytes = safetensors::tensor::serialize(vec![("w".to_string(), view)], None).unwrap();

    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    assert_eq!(loaded["w"].dtype(), DType::BF16);

    // Load into VarBuilder requesting BF16 (no conversion needed).
    let vb = VarBuilder::from_tensors(loaded, DType::BF16, &Device::Cpu);
    let t = vb.get(&[4], "w").unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    let f32_vals = t.to_f32_array().unwrap();
    let vals: Vec<f32> = f32_vals.iter().copied().collect();
    assert!((vals[0] - 1.0).abs() < 0.1);
    assert!((vals[1] - (-2.5)).abs() < 0.1);
}

#[test]
fn test_f16_safetensors_into_varbuilder_with_upcast() {
    // Build F16 safetensors, load into F32 VarBuilder (upcast).
    let values = [0.5f32, 1.5, -2.0];
    let f16_bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect();

    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![3], &f16_bytes).unwrap();
    let bytes = safetensors::tensor::serialize(vec![("w".to_string(), view)], None).unwrap();

    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    assert_eq!(loaded["w"].dtype(), DType::F16);

    // Load into F32 VarBuilder: should upcast from F16 to F32.
    let vb = VarBuilder::from_tensors(loaded, DType::F32, &Device::Cpu);
    let t = vb.get(&[3], "w").unwrap();
    assert_eq!(t.dtype(), DType::F32);
    let data = t.to_flat_vec::<f32>().unwrap();
    assert!((data[0] - 0.5).abs() < 0.01);
    assert!((data[1] - 1.5).abs() < 0.01);
    assert!((data[2] - (-2.0)).abs() < 0.01);
}
