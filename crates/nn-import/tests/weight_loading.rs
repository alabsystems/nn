// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for safetensors weight loading and mapping pipeline.
//!
//! Covers: header parsing, dtype conversion, shape validation, name mapping,
//! multi-file loading, missing weight detection, weight permutation, and
//! zero-weight initialization.
//!
//! Part of #3942.

use std::collections::HashMap;

use nn_core::dyn_tensor::DynTensor;
use nn_core::var_builder::{HfToNnMapper, VarBuilder, WeightNameMapper};
use nn_core::{DType, Device};
use nn_import::{
    build_graph, build_weight_map, kokoro_name_mapping, map_pytorch_key, parse_exported_program,
    validate_kokoro_keys, validate_kokoro_safetensors, ImportError, ResolvedWeight,
};

// =============================================================================
// Test helpers
// =============================================================================

/// Build a safetensors byte buffer from typed tensor data.
fn build_safetensors_f32(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
    let byte_bufs: Vec<Vec<u8>> = tensors
        .iter()
        .map(|&(_, _, data)| data.iter().flat_map(|v| v.to_le_bytes()).collect())
        .collect();
    let mut tensor_map: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();
    for (i, &(name, shape, _)) in tensors.iter().enumerate() {
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            shape.to_vec(),
            &byte_bufs[i],
        )
        .expect("valid tensor view");
        tensor_map.push((name.to_string(), view));
    }
    safetensors::tensor::serialize(tensor_map, None).expect("serialization should succeed")
}

/// Build a safetensors byte buffer with a specified dtype.
fn build_safetensors_typed(tensors: &[(&str, &[usize], &[u8], safetensors::Dtype)]) -> Vec<u8> {
    let mut tensor_map: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();
    for &(name, shape, data, dtype) in tensors {
        let view = safetensors::tensor::TensorView::new(dtype, shape.to_vec(), data)
            .expect("valid tensor view");
        tensor_map.push((name.to_string(), view));
    }
    safetensors::tensor::serialize(tensor_map, None).expect("serialization should succeed")
}

/// Minimal linear model JSON for graph building tests.
fn minimal_linear_json() -> &'static str {
    r#"{
        "graph_module": {
            "graph": {
                "inputs": [
                    {"as_tensor": {"name": "p_weight"}},
                    {"as_tensor": {"name": "p_bias"}},
                    {"as_tensor": {"name": "x"}}
                ],
                "outputs": [{"as_tensor": {"name": "linear"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.linear.default",
                        "inputs": [
                            {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
                            {"name": "weight", "arg": {"as_tensor": {"name": "p_weight"}}, "kind": 1},
                            {"name": "bias", "arg": {"as_tensor": {"name": "p_bias"}}, "kind": 1}
                        ],
                        "outputs": [{"as_tensor": {"name": "linear"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "x": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]},
                    "p_weight": {"dtype": 7, "sizes": [{"as_int": 3}, {"as_int": 4}], "requires_grad": true, "strides": [{"as_int": 4}, {"as_int": 1}]},
                    "p_bias": {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": true, "strides": [{"as_int": 1}]},
                    "linear": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 3}, {"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_weight"}, "parameter_name": "weight"}},
                    {"parameter": {"arg": {"name": "p_bias"}, "parameter_name": "bias"}},
                    {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "linear"}}}}
                ]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "opset_version": {"aten": 10},
        "range_constraints": {}
    }"#
}

// =============================================================================
// 1. Safetensors header parsing
// =============================================================================

#[test]
fn test_safetensors_header_parse_single_tensor() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes = build_safetensors_f32(&[("layer.weight", &[2, 3], &data)]);

    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let view = tensors.tensor("layer.weight").unwrap();

    assert_eq!(view.shape(), &[2, 3]);
    assert_eq!(view.dtype(), safetensors::Dtype::F32);
    assert_eq!(view.data().len(), 24); // 6 * 4 bytes
}

#[test]
fn test_safetensors_header_parse_multiple_tensors() {
    let w: Vec<f32> = vec![1.0; 12];
    let b: Vec<f32> = vec![0.0; 3];
    let bytes = build_safetensors_f32(&[
        ("model.linear.weight", &[3, 4], &w),
        ("model.linear.bias", &[3], &b),
    ]);

    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let names: Vec<String> = tensors.names().into_iter().map(String::from).collect();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"model.linear.weight".to_string()));
    assert!(names.contains(&"model.linear.bias".to_string()));

    let weight_view = tensors.tensor("model.linear.weight").unwrap();
    assert_eq!(weight_view.shape(), &[3, 4]);

    let bias_view = tensors.tensor("model.linear.bias").unwrap();
    assert_eq!(bias_view.shape(), &[3]);
}

#[test]
fn test_safetensors_header_parse_empty() {
    let bytes = build_safetensors_f32(&[]);
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    assert_eq!(tensors.names().len(), 0);
}

#[test]
fn test_safetensors_header_parse_scalar_tensor() {
    let data: Vec<f32> = vec![42.0];
    let bytes = build_safetensors_f32(&[("scalar_param", &[], &data)]);

    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let view = tensors.tensor("scalar_param").unwrap();
    assert_eq!(view.shape(), &[] as &[usize]);
    assert_eq!(view.data().len(), 4); // 1 * 4 bytes
}

#[test]
fn test_safetensors_header_parse_high_rank_tensor() {
    let data: Vec<f32> = vec![0.1; 2 * 3 * 4 * 5];
    let bytes = build_safetensors_f32(&[("4d_tensor", &[2, 3, 4, 5], &data)]);

    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let view = tensors.tensor("4d_tensor").unwrap();
    assert_eq!(view.shape(), &[2, 3, 4, 5]);
}

#[test]
fn test_safetensors_invalid_bytes_returns_error() {
    let result = safetensors::SafeTensors::deserialize(&[0u8; 16]);
    assert!(result.is_err(), "invalid bytes should fail to parse");
}

// =============================================================================
// 2. Weight name mapping: PyTorch -> nn
// =============================================================================

#[test]
fn test_pytorch_to_nn_identity_mapping_kokoro() {
    // Kokoro uses identity mapping — PyTorch names match nn names.
    let keys = [
        "plbert.embeddings.word_embeddings.weight",
        "plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight",
        "bert_encoder.weight",
        "text_encoder.lstm.weight_ih_l0",
        "prosody_predictor.shared.0.conv.weight",
        "predictor.F0.0.n1.fc.weight",
        "decoder.conv_pre.weight",
        "decoder.resblocks.0.convs1.0.weight",
    ];
    for key in &keys {
        let mapped = map_pytorch_key(key);
        assert_eq!(
            mapped.as_deref(),
            Some(*key),
            "identity mapping failed for {key}"
        );
    }
}

#[test]
fn test_pytorch_to_nn_unknown_prefix_returns_none() {
    assert_eq!(map_pytorch_key("unknown_module.weight"), None);
    assert_eq!(map_pytorch_key(""), None);
    assert_eq!(map_pytorch_key("model.encoder.weight"), None);
}

#[test]
fn test_hf_to_nn_transformer_decoder_mapping() {
    let mapper = HfToNnMapper::decoder_transformer();

    // Standard HF transformer names -> nn names
    let pairs = [
        (
            "layers.0.attn.q.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ),
        (
            "layers.0.attn.v.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ),
        (
            "layers.0.mlp.gate.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ),
        (
            "layers.0.ln1.weight",
            "model.layers.0.input_layernorm.weight",
        ),
    ];
    for (nn_name, expected_hf) in &pairs {
        assert_eq!(
            mapper.map_name(nn_name),
            *expected_hf,
            "mapper failed for {nn_name}"
        );
    }
}

#[test]
fn test_hf_to_nn_with_layer_indices() {
    let mapper = HfToNnMapper::decoder_transformer();

    // Verify multiple layer indices map correctly.
    for i in 0..5 {
        let nn = format!("layers.{i}.attn.q.weight");
        let hf = format!("model.layers.{i}.self_attn.q_proj.weight");
        assert_eq!(mapper.map_name(&nn), hf, "layer index {i} mapping failed");
    }
}

#[test]
fn test_hf_to_nn_prefix_only_mapping() {
    let mapper = HfToNnMapper::siglip2_granite_docling();

    assert_eq!(
        mapper.map_name("encoder.layers.0.weight"),
        "model.vision_model.encoder.layers.0.weight"
    );
}

#[test]
fn test_name_mapping_closure_integration() {
    let closure = kokoro_name_mapping();

    // Known keys pass through.
    assert_eq!(
        closure("decoder.conv_pre.weight"),
        "decoder.conv_pre.weight"
    );
    // Unknown keys also pass through (fallback).
    assert_eq!(closure("unknown.key"), "unknown.key");
}

#[test]
fn test_verify_mapper_coverage_all_found() {
    let mapper = HfToNnMapper::new().with_prefix_rule("model", "m");

    let checkpoint_names = vec!["model.weight".to_string(), "model.bias".to_string()];
    let nn_names = vec!["m.weight".to_string(), "m.bias".to_string()];

    let missing =
        nn_core::var_builder::verify_mapper_coverage(&nn_names, &checkpoint_names, &mapper);
    assert!(
        missing.is_empty(),
        "all names should be found, got: {missing:?}"
    );
}

#[test]
fn test_verify_mapper_coverage_detects_missing() {
    let mapper = HfToNnMapper::new().with_prefix_rule("model", "m");

    let checkpoint_names = vec!["model.weight".to_string()];
    let nn_names = vec!["m.weight".to_string(), "m.extra".to_string()];

    let missing =
        nn_core::var_builder::verify_mapper_coverage(&nn_names, &checkpoint_names, &mapper);
    assert_eq!(missing, vec!["m.extra"]);
}

// =============================================================================
// 3. Shape validation
// =============================================================================

#[test]
fn test_shape_validation_correct_dims() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".to_string(),
        DynTensor::new(&[1.0; 12], &[3, 4], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    let t = vb.get(&[3, 4], "weight").unwrap();
    assert_eq!(t.dims(), &[3, 4]);
}

#[test]
fn test_shape_validation_wrong_dims_returns_error() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".to_string(),
        DynTensor::new(&[1.0; 12], &[3, 4], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    let err = vb.get(&[4, 3], "weight").unwrap_err();
    match err {
        nn_core::TensorError::ShapeMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, vec![4, 3]);
            assert_eq!(actual, vec![3, 4]);
        }
        other => panic!("expected ShapeMismatch, got: {other:?}"),
    }
}

#[test]
fn test_shape_validation_wrong_rank_returns_error() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".to_string(),
        DynTensor::new(&[1.0; 12], &[3, 4], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    // Request rank-1 but stored is rank-2.
    let err = vb.get(&[12], "weight").unwrap_err();
    assert!(
        matches!(err, nn_core::TensorError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got: {err:?}"
    );
}

#[test]
fn test_shape_validation_via_build_weight_map() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();

    let mut weights = HashMap::new();
    weights.insert("weight".to_string(), (vec![0.1; 12], vec![3, 4]));
    weights.insert("bias".to_string(), (vec![0.0; 3], vec![3]));

    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weights);

    // Verify shapes from the weight map match what we put in.
    assert_eq!(weight_map["p_weight"].shape, vec![3, 4]);
    assert_eq!(weight_map["p_weight"].data.len(), 12);
    assert_eq!(weight_map["p_bias"].shape, vec![3]);
    assert_eq!(weight_map["p_bias"].data.len(), 3);
}

#[test]
fn test_weight_shape_mismatch_data_vs_shape() {
    // ResolvedWeight with data length mismatching shape product.
    let weight = ResolvedWeight::new(vec![0.1; 10], vec![3, 4]); // 10 != 3*4=12
    assert_eq!(weight.data.len(), 10);
    assert_eq!(weight.shape, vec![3, 4]);
    // The shape/data mismatch is detected at graph execution time, not at weight_map construction.
    // Here we verify the data is stored as-is (no implicit validation at construction).
}

// =============================================================================
// 4. Dtype conversion: F16/BF16/F64/I64/U8/I8 -> F32
// =============================================================================

#[test]
fn test_dtype_conversion_f32_passthrough() {
    let data: Vec<f32> = vec![1.0, -2.5, 0.0, 3.14];
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let st_bytes = build_safetensors_typed(&[("w", &[4], &bytes, safetensors::Dtype::F32)]);
    let tensors = safetensors::SafeTensors::deserialize(&st_bytes).unwrap();
    let view = tensors.tensor("w").unwrap();

    let f32_data: Vec<f32> = view
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(f32_data.len(), 4);
    assert!((f32_data[0] - 1.0).abs() < f32::EPSILON);
    assert!((f32_data[1] + 2.5).abs() < f32::EPSILON);
    assert!((f32_data[3] - 3.14).abs() < 0.001);
}

#[test]
fn test_dtype_conversion_f16_to_f32() {
    let f16_vals: Vec<half::f16> = vec![
        half::f16::from_f32(1.0),
        half::f16::from_f32(-0.5),
        half::f16::from_f32(0.0),
        half::f16::from_f32(65504.0), // f16 max
    ];
    let bytes: Vec<u8> = f16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

    let st_bytes = build_safetensors_typed(&[("w", &[4], &bytes, safetensors::Dtype::F16)]);
    let tensors = safetensors::SafeTensors::deserialize(&st_bytes).unwrap();
    let view = tensors.tensor("w").unwrap();

    let f32_data: Vec<f32> = view
        .data()
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();
    assert_eq!(f32_data.len(), 4);
    assert!((f32_data[0] - 1.0).abs() < 0.01);
    assert!((f32_data[1] + 0.5).abs() < 0.01);
    assert_eq!(f32_data[2], 0.0);
    assert!((f32_data[3] - 65504.0).abs() < 1.0);
}

#[test]
fn test_dtype_conversion_bf16_to_f32() {
    let bf16_vals: Vec<half::bf16> = vec![
        half::bf16::from_f32(1.0),
        half::bf16::from_f32(-3.0),
        half::bf16::from_f32(0.001),
    ];
    let bytes: Vec<u8> = bf16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

    let st_bytes = build_safetensors_typed(&[("w", &[3], &bytes, safetensors::Dtype::BF16)]);
    let tensors = safetensors::SafeTensors::deserialize(&st_bytes).unwrap();
    let view = tensors.tensor("w").unwrap();

    let f32_data: Vec<f32> = view
        .data()
        .chunks_exact(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();
    assert_eq!(f32_data.len(), 3);
    assert!((f32_data[0] - 1.0).abs() < 0.02);
    assert!((f32_data[1] + 3.0).abs() < 0.1);
    assert!((f32_data[2] - 0.001).abs() < 0.001);
}

#[test]
fn test_dtype_conversion_f64_to_f32() {
    let f64_vals: Vec<f64> = vec![1.0, -2.5, 0.0, 1e30];
    let bytes: Vec<u8> = f64_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

    let st_bytes = build_safetensors_typed(&[("w", &[4], &bytes, safetensors::Dtype::F64)]);
    let tensors = safetensors::SafeTensors::deserialize(&st_bytes).unwrap();
    let view = tensors.tensor("w").unwrap();

    let f32_data: Vec<f32> = view
        .data()
        .chunks_exact(8)
        .map(|c| {
            let arr: [u8; 8] = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
            f64::from_le_bytes(arr) as f32
        })
        .collect();
    assert_eq!(f32_data.len(), 4);
    assert!((f32_data[0] - 1.0).abs() < f32::EPSILON);
    assert!((f32_data[1] + 2.5).abs() < f32::EPSILON);
}

#[test]
fn test_dtype_conversion_i64_to_f32() {
    let i64_vals: Vec<i64> = vec![0, 1, -1, 42, i64::MAX];
    let bytes: Vec<u8> = i64_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

    let st_bytes = build_safetensors_typed(&[("w", &[5], &bytes, safetensors::Dtype::I64)]);
    let tensors = safetensors::SafeTensors::deserialize(&st_bytes).unwrap();
    let view = tensors.tensor("w").unwrap();

    let f32_data: Vec<f32> = view
        .data()
        .chunks_exact(8)
        .map(|c| {
            let arr: [u8; 8] = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
            i64::from_le_bytes(arr) as f32
        })
        .collect();
    assert_eq!(f32_data.len(), 5);
    assert_eq!(f32_data[0], 0.0);
    assert_eq!(f32_data[1], 1.0);
    assert_eq!(f32_data[2], -1.0);
    assert_eq!(f32_data[3], 42.0);
}

#[test]
fn test_dtype_conversion_u8_to_f32() {
    let u8_vals: Vec<u8> = vec![0, 1, 127, 255];

    let st_bytes = build_safetensors_typed(&[("w", &[4], &u8_vals, safetensors::Dtype::U8)]);
    let tensors = safetensors::SafeTensors::deserialize(&st_bytes).unwrap();
    let view = tensors.tensor("w").unwrap();

    let f32_data: Vec<f32> = view.data().iter().map(|&b| f32::from(b)).collect();
    assert_eq!(f32_data, vec![0.0, 1.0, 127.0, 255.0]);
}

#[test]
fn test_dtype_conversion_i8_to_f32() {
    let i8_vals: Vec<i8> = vec![0, 1, -1, 127, -128];
    let bytes: Vec<u8> = i8_vals.iter().map(|&v| v as u8).collect();

    let st_bytes = build_safetensors_typed(&[("w", &[5], &bytes, safetensors::Dtype::I8)]);
    let tensors = safetensors::SafeTensors::deserialize(&st_bytes).unwrap();
    let view = tensors.tensor("w").unwrap();

    let f32_data: Vec<f32> = view.data().iter().map(|&b| f32::from(b as i8)).collect();
    assert_eq!(f32_data, vec![0.0, 1.0, -1.0, 127.0, -128.0]);
}

#[test]
fn test_dtype_conversion_f16_precision_loss_within_tolerance() {
    // Verify that f16 -> f32 conversion stays within expected precision bounds.
    let original: f32 = 0.123456;
    let f16_val = half::f16::from_f32(original);
    let roundtrip = f16_val.to_f32();

    // f16 has ~3 decimal digits of precision.
    let error = (original - roundtrip).abs();
    assert!(
        error < 0.001,
        "f16 roundtrip error {error} exceeds 0.001 tolerance"
    );
}

#[test]
fn test_dtype_conversion_bf16_precision_loss_within_tolerance() {
    // bf16 has ~2 decimal digits of precision but wider range than f16.
    let original: f32 = 1234.5;
    let bf16_val = half::bf16::from_f32(original);
    let roundtrip = bf16_val.to_f32();

    let error = (original - roundtrip).abs();
    assert!(
        error < 10.0,
        "bf16 roundtrip error {error} exceeds 10.0 tolerance for value {original}"
    );
}

// =============================================================================
// 5. Multi-file loading (simulated via separate buffers)
// =============================================================================

#[test]
fn test_multi_file_loading_merge_tensors() {
    // Simulate two safetensors files (model split across shards).
    let shard1 = build_safetensors_f32(&[
        ("model.layer0.weight", &[4, 4], &[1.0; 16]),
        ("model.layer0.bias", &[4], &[0.1; 4]),
    ]);
    let shard2 = build_safetensors_f32(&[
        ("model.layer1.weight", &[2, 4], &[2.0; 8]),
        ("model.layer1.bias", &[2], &[0.2; 2]),
    ]);

    // Load each shard and merge into a single tensor map.
    let mut all_tensors: HashMap<String, DynTensor> = HashMap::new();

    for bytes in &[&shard1, &shard2] {
        let tensors = safetensors::SafeTensors::deserialize(bytes).unwrap();
        for name in tensors.names() {
            let view = tensors.tensor(name).unwrap();
            let shape: Vec<usize> = view.shape().to_vec();
            let f32_data: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            all_tensors.insert(
                name.to_string(),
                DynTensor::new(&f32_data, &shape, &Device::Cpu).unwrap(),
            );
        }
    }

    let vb = VarBuilder::from_tensors(all_tensors, DType::F32, &Device::Cpu);

    // Both shards' tensors should be accessible.
    let w0 = vb.pp("model").pp("layer0").get(&[4, 4], "weight").unwrap();
    assert_eq!(w0.dims(), &[4, 4]);

    let w1 = vb.pp("model").pp("layer1").get(&[2, 4], "weight").unwrap();
    assert_eq!(w1.dims(), &[2, 4]);

    let b0 = vb.pp("model").pp("layer0").get(&[4], "bias").unwrap();
    assert_eq!(b0.to_flat_vec::<f32>().unwrap(), vec![0.1; 4]);

    let b1 = vb.pp("model").pp("layer1").get(&[2], "bias").unwrap();
    assert_eq!(b1.to_flat_vec::<f32>().unwrap(), vec![0.2; 2]);
}

#[test]
fn test_multi_file_loading_duplicate_key_last_wins() {
    // When the same key appears in two shards, last insertion wins.
    let shard1 = build_safetensors_f32(&[("shared.weight", &[2], &[1.0, 2.0])]);
    let shard2 = build_safetensors_f32(&[("shared.weight", &[2], &[3.0, 4.0])]);

    let mut all_tensors: HashMap<String, DynTensor> = HashMap::new();
    for bytes in &[&shard1, &shard2] {
        let tensors = safetensors::SafeTensors::deserialize(bytes).unwrap();
        for name in tensors.names() {
            let view = tensors.tensor(name).unwrap();
            let shape: Vec<usize> = view.shape().to_vec();
            let f32_data: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            all_tensors.insert(
                name.to_string(),
                DynTensor::new(&f32_data, &shape, &Device::Cpu).unwrap(),
            );
        }
    }

    let vb = VarBuilder::from_tensors(all_tensors, DType::F32, &Device::Cpu);
    let t = vb.get(&[2], "shared.weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![3.0, 4.0]); // shard2 wins
}

#[test]
fn test_multi_file_loading_tensor_count() {
    let shard1 = build_safetensors_f32(&[("a", &[1], &[1.0]), ("b", &[1], &[2.0])]);
    let shard2 = build_safetensors_f32(&[("c", &[1], &[3.0]), ("d", &[1], &[4.0])]);

    let mut all_tensors: HashMap<String, DynTensor> = HashMap::new();
    for bytes in &[&shard1, &shard2] {
        let tensors = safetensors::SafeTensors::deserialize(bytes).unwrap();
        for name in tensors.names() {
            let view = tensors.tensor(name).unwrap();
            let f32_data: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            all_tensors.insert(
                name.to_string(),
                DynTensor::new(&f32_data, view.shape().to_vec(), &Device::Cpu).unwrap(),
            );
        }
    }

    assert_eq!(all_tensors.len(), 4);
}

// =============================================================================
// 6. Missing weight detection
// =============================================================================

#[test]
fn test_missing_weight_in_graph_build() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();

    let err = build_graph(&program, &empty_weights).unwrap_err();
    assert!(
        matches!(err, ImportError::MissingWeight { .. }),
        "expected MissingWeight, got: {err:?}"
    );
}

#[test]
fn test_missing_weight_partial_weights() {
    use nn_core::dyn_tensor::trace::TraceOp;

    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();

    // Provide only weight, not bias.
    let mut weight_data = HashMap::new();
    weight_data.insert("weight".to_string(), (vec![0.1; 12], vec![3, 4]));
    // No bias entry.

    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);

    // build_weight_map silently omits missing weights.
    assert!(weight_map.contains_key("p_weight"));
    assert!(!weight_map.contains_key("p_bias"));

    // bias is an OPTIONAL weight for Linear (see op_map_impl::map_linear, which
    // uses optional_weight for bias and resolve_weight only for the required
    // weight). With the required weight present and bias absent, the graph
    // builds successfully and the Linear op carries `bias: None`.
    let imported = build_graph(&program, &weight_map).unwrap();
    let linear = imported
        .graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Linear { .. }))
        .expect("graph should contain a Linear op");
    match linear.op() {
        TraceOp::Linear { weight, bias } => {
            assert_eq!(weight.shape(), &[3, 4], "weight must be loaded");
            assert!(bias.is_none(), "missing bias must resolve to None");
        }
        other => panic!("expected Linear, got: {other:?}"),
    }
}

#[test]
fn test_missing_weight_via_varbuilder() {
    let tensors = HashMap::new();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    let err = vb.pp("model").get(&[3, 4], "weight").unwrap_err();
    assert!(
        matches!(err, nn_core::TensorError::TensorNotFound { .. }),
        "expected TensorNotFound, got: {err:?}"
    );
}

#[test]
fn test_missing_kokoro_weight_groups() {
    let keys: Vec<String> = vec!["plbert.x".to_string()];
    let result = validate_kokoro_safetensors(&keys);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("bert_encoder."),
        "should report bert_encoder missing"
    );
    assert!(msg.contains("decoder."), "should report decoder missing");
}

#[test]
fn test_validate_kokoro_keys_reports_all_missing() {
    let keys: Vec<&str> = vec![];
    let missing = validate_kokoro_keys(&keys);
    assert_eq!(
        missing.len(),
        6,
        "all 6 prefixes should be missing for empty keys"
    );
}

#[test]
fn test_contains_tensor_with_missing_key() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "encoder.weight".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    assert!(vb.pp("encoder").contains_tensor("weight"));
    assert!(!vb.pp("encoder").contains_tensor("bias"));
    assert!(!vb.pp("decoder").contains_tensor("weight"));
}

// =============================================================================
// 7. Weight permutation (transpose/reshape)
// =============================================================================

#[test]
fn test_weight_transpose_for_linear_layer() {
    // PyTorch Linear stores weight as [out_features, in_features].
    // If nn expected [in_features, out_features], we'd need to transpose.
    // nn matches PyTorch convention, so this test verifies the data layout.
    let weight_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, // row 0 (out_feature 0)
        4.0, 5.0, 6.0, // row 1 (out_feature 1)
    ];
    let weight = ResolvedWeight::new(weight_data.clone(), vec![2, 3]);
    assert_eq!(weight.shape, vec![2, 3]);
    assert_eq!(weight.data, weight_data);

    // Simulate transpose if needed: reshape [2,3] data into [3,2].
    let rows = 2usize;
    let cols = 3usize;
    let mut transposed = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = weight_data[r * cols + c];
        }
    }
    assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let transposed_weight = ResolvedWeight::new(transposed, vec![3, 2]);
    assert_eq!(transposed_weight.shape, vec![3, 2]);
}

#[test]
fn test_weight_reshape_conv_to_linear() {
    // Some models reshape conv1x1 weights [C_out, C_in, 1, 1] to linear [C_out, C_in].
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let conv_weight = ResolvedWeight::new(data.clone(), vec![3, 2, 1, 1]);
    assert_eq!(conv_weight.shape, vec![3, 2, 1, 1]);

    // Reshape: [3, 2, 1, 1] -> [3, 2] (squeeze trailing 1s).
    let product: usize = conv_weight.shape.iter().product();
    let squeezed_shape: Vec<usize> = conv_weight
        .shape
        .iter()
        .copied()
        .filter(|&d| d > 1)
        .collect();
    assert_eq!(squeezed_shape, vec![3, 2]);
    assert_eq!(product, data.len());

    let reshaped = ResolvedWeight::new(data, squeezed_shape);
    assert_eq!(reshaped.shape, vec![3, 2]);
}

#[test]
fn test_weight_reshape_flat_to_multi_dim() {
    // Some checkpoints store weights flat; reshape to expected dims.
    let flat_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let _flat_weight = ResolvedWeight::new(flat_data.clone(), vec![24]);

    // Reshape to [2, 3, 4].
    let target_shape = vec![2, 3, 4];
    let target_numel: usize = target_shape.iter().product();
    assert_eq!(flat_data.len(), target_numel);

    let reshaped = ResolvedWeight::new(flat_data, target_shape.clone());
    assert_eq!(reshaped.shape, target_shape);
    assert_eq!(reshaped.data[0], 0.0);
    assert_eq!(reshaped.data[23], 23.0);
}

#[test]
fn test_weight_dyntensor_reshape() {
    // DynTensor reshape for weight manipulation.
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[3, 4], &Device::Cpu).unwrap();
    let reshaped = t.reshape([4, 3]).unwrap();
    assert_eq!(reshaped.dims(), &[4, 3]);
    assert_eq!(reshaped.to_flat_vec::<f32>().unwrap().len(), 12);
}

#[test]
fn test_weight_dyntensor_transpose() {
    // DynTensor transpose for weight permutation.
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = DynTensor::new(&data, &[2, 3], &Device::Cpu).unwrap();
    let transposed = t.t().unwrap();
    assert_eq!(transposed.dims(), &[3, 2]);

    let flat = transposed.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

// =============================================================================
// 8. Zero-weight initialization
// =============================================================================

#[test]
fn test_zeros_backend_provides_zero_weights() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);

    let w = vb.pp("encoder").get(&[3, 4], "weight").unwrap();
    assert_eq!(w.dims(), &[3, 4]);
    let data = w.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| v == 0.0), "all values should be zero");
}

#[test]
fn test_zeros_backend_any_name_succeeds() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);

    // Any path/name combination should succeed with zeros.
    let t1 = vb
        .pp("model")
        .pp("layer.0")
        .get(&[512, 768], "weight")
        .unwrap();
    assert_eq!(t1.dims(), &[512, 768]);

    let t2 = vb.pp("decoder").get(&[1], "bias").unwrap();
    assert_eq!(t2.dims(), &[1]);
}

#[test]
fn test_zeros_backend_contains_always_true() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    assert!(vb.contains_tensor("anything"));
    assert!(vb.pp("deep").pp("path").contains_tensor("weight"));
}

#[test]
fn test_optional_weight_fallback_to_zeros() {
    // Pattern: Try loading from tensor map, fallback to zeros for optional weights.
    let mut tensors = HashMap::new();
    tensors.insert(
        "encoder.weight".to_string(),
        DynTensor::new(&[1.0; 12], &[3, 4], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    // Required weight: exists.
    let w = vb.pp("encoder").get(&[3, 4], "weight").unwrap();
    assert_eq!(w.dims(), &[3, 4]);

    // Optional weight: does not exist, fallback to zeros.
    let bias_result = vb.pp("encoder").get(&[3], "bias");
    let bias = if bias_result.is_ok() {
        bias_result.unwrap()
    } else {
        DynTensor::zeros(&[3], DType::F32, &Device::Cpu).unwrap()
    };
    assert_eq!(bias.dims(), &[3]);
    assert!(
        bias.to_flat_vec::<f32>().unwrap().iter().all(|&v| v == 0.0),
        "fallback bias should be zeros"
    );
}

#[test]
fn test_zero_initialized_weight_with_specific_dtype() {
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let t = vb.get(&[2, 3], "weight").unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    assert_eq!(t.dims(), &[2, 3]);
}

// =============================================================================
// End-to-end: Full pipeline weight loading round-trip
// =============================================================================

#[test]
fn test_e2e_safetensors_to_varbuilder_round_trip() {
    // 1. Create safetensors data.
    let weight_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
    let bias_data: Vec<f32> = vec![0.01, 0.02, 0.03];

    let bytes = build_safetensors_f32(&[
        ("encoder.weight", &[3, 4], &weight_data),
        ("encoder.bias", &[3], &bias_data),
    ]);

    // 2. Parse safetensors into tensor map.
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut tensor_map: HashMap<String, DynTensor> = HashMap::new();
    for name in tensors.names() {
        let view = tensors.tensor(name).unwrap();
        let shape: Vec<usize> = view.shape().to_vec();
        let f32_data: Vec<f32> = view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        tensor_map.insert(
            name.to_string(),
            DynTensor::new(&f32_data, &shape, &Device::Cpu).unwrap(),
        );
    }

    // 3. Create VarBuilder and load.
    let vb = VarBuilder::from_tensors(tensor_map, DType::F32, &Device::Cpu);

    let w = vb.pp("encoder").get(&[3, 4], "weight").unwrap();
    let b = vb.pp("encoder").get(&[3], "bias").unwrap();

    assert_eq!(w.dims(), &[3, 4]);
    assert_eq!(b.dims(), &[3]);

    let w_flat = w.to_flat_vec::<f32>().unwrap();
    assert!((w_flat[0] - 0.0).abs() < f32::EPSILON);
    assert!((w_flat[1] - 0.1).abs() < f32::EPSILON);
    assert!((w_flat[11] - 1.1).abs() < f32::EPSILON);

    let b_flat = b.to_flat_vec::<f32>().unwrap();
    assert!((b_flat[0] - 0.01).abs() < f32::EPSILON);
    assert!((b_flat[2] - 0.03).abs() < f32::EPSILON);
}

#[test]
fn test_e2e_safetensors_with_name_mapping() {
    // Safetensors uses HF names, model code uses NN names.
    let weight_data: Vec<f32> = vec![1.0; 8];
    let bytes = build_safetensors_f32(&[(
        "model.layers.0.self_attn.q_proj.weight",
        &[2, 4],
        &weight_data,
    )]);

    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut tensor_map: HashMap<String, DynTensor> = HashMap::new();
    for name in tensors.names() {
        let view = tensors.tensor(name).unwrap();
        let shape: Vec<usize> = view.shape().to_vec();
        let f32_data: Vec<f32> = view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        tensor_map.insert(
            name.to_string(),
            DynTensor::new(&f32_data, &shape, &Device::Cpu).unwrap(),
        );
    }

    // Apply decoder_transformer mapper (NN short names -> HF names).
    let mapper = HfToNnMapper::decoder_transformer();
    let vb = VarBuilder::from_tensors(tensor_map, DType::F32, &Device::Cpu)
        .with_weight_name_mapper(mapper);

    // NN model code uses short names.
    let t = vb
        .pp("layers")
        .pp("0")
        .pp("attn")
        .get(&[2, 4], "q.weight")
        .unwrap();
    assert_eq!(t.dims(), &[2, 4]);
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0; 8]);
}

#[test]
fn test_e2e_graph_build_with_safetensors_data() {
    // Build a graph using weights loaded from in-memory safetensors.
    let weight_vals: Vec<f32> = vec![0.1; 12];
    let bias_vals: Vec<f32> = vec![0.0; 3];

    let bytes = build_safetensors_f32(&[
        ("weight", &[3, 4], &weight_vals),
        ("bias", &[3], &bias_vals),
    ]);

    // Parse safetensors into the weight format expected by build_weight_map.
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for name in tensors.names() {
        let view = tensors.tensor(name).unwrap();
        let shape: Vec<usize> = view.shape().to_vec();
        let f32_data: Vec<f32> = view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        weight_data.insert(name.to_string(), (f32_data, shape));
    }

    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);

    let imported = build_graph(&program, &weight_map).unwrap();
    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.output_names, vec!["linear"]);
}

// =============================================================================
// Edge cases: NaN/Inf rejection in weight loading
// =============================================================================

#[test]
fn test_nan_weight_rejected_by_varbuilder() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, f32::NAN, 3.0], &[3], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    let err = vb.get(&[3], "w").unwrap_err();
    assert!(
        matches!(err, nn_core::TensorError::NonFiniteData { .. }),
        "expected NonFiniteData, got: {err:?}"
    );
}

#[test]
fn test_inf_weight_rejected_by_varbuilder() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "w".to_string(),
        DynTensor::new(&[f32::INFINITY, f32::NEG_INFINITY], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    let err = vb.get(&[2], "w").unwrap_err();
    match err {
        nn_core::TensorError::NonFiniteData { name, count } => {
            assert_eq!(name, "w");
            assert_eq!(count, 2);
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

#[test]
fn test_finite_weights_pass_validation() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, -2.5, 0.0, 1e30], &[4], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    let t = vb.get(&[4], "w").unwrap();
    assert_eq!(t.dims(), &[4]);
}
