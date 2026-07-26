// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended VarBuilder weight loading tests.
//!
//! Covers hierarchy (push_prefix nested scope), weight name mapping
//! (HfToNnMapper), ZerosBackend, TensorMapBackend, safetensors
//! round-trip through VarBuilder, error cases, WeightNameMapper
//! coverage, and mixed-dtype (F32, F16, BF16) loading.
//!
//! Part of #4186.

use std::collections::HashMap;
use std::sync::Arc;

use crate::dyn_tensor::{
    load_safetensors, load_safetensors_from_bytes, save_safetensors, tensors_to_safetensors_bytes,
    DynTensor,
};
use crate::var_builder::{
    verify_mapper_coverage, HfToNnMapper, TensorBackend, TensorMapBackend, VarBuilder,
    WeightNameMapper, ZerosBackend,
};
use crate::{DType, Device, TensorError};

// ===========================================================================
// 1. VarBuilder hierarchy -- push_prefix creates nested scope, path joins
// ===========================================================================

#[test]
fn test_hierarchy_root_prefix_is_empty() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    assert_eq!(vb.prefix(), "");
}

#[test]
fn test_hierarchy_single_pp_creates_one_level() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let child = vb.pp("encoder");
    assert_eq!(child.prefix(), "encoder");
}

#[test]
fn test_hierarchy_chained_pp_joins_with_dots() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let deep = vb.pp("model").pp("layers").pp("0").pp("self_attn");
    assert_eq!(deep.prefix(), "model.layers.0.self_attn");
}

#[test]
fn test_hierarchy_pp_with_dotted_segment() {
    // A single pp() call can contain dots, which become part of one segment.
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let child = vb.pp("encoder.layer.0");
    assert_eq!(child.prefix(), "encoder.layer.0");

    // Adding another level:
    let grandchild = child.pp("weight");
    assert_eq!(grandchild.prefix(), "encoder.layer.0.weight");
}

#[test]
fn test_hierarchy_empty_pp_segments_are_skipped() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let child = vb.pp("").pp("a").pp("").pp("b").pp("");
    assert_eq!(child.prefix(), "a.b");
}

#[test]
fn test_hierarchy_pp_does_not_mutate_parent() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let _child = vb.pp("child");
    assert_eq!(vb.prefix(), "");

    let parent = vb.pp("parent");
    let _grandchild = parent.pp("grandchild");
    assert_eq!(parent.prefix(), "parent");
}

#[test]
fn test_hierarchy_resolves_tensor_key_through_pp() {
    let mut map = HashMap::new();
    map.insert(
        "model.encoder.layers.0.attn.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);

    let t = vb
        .pp("model")
        .pp("encoder")
        .pp("layers")
        .pp("0")
        .pp("attn")
        .get(&[2, 2], "weight")
        .unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_hierarchy_contains_tensor_at_correct_level_only() {
    let mut map = HashMap::new();
    map.insert(
        "a.b.weight".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);

    // Correct path:
    assert!(vb.pp("a").pp("b").contains_tensor("weight"));
    // Wrong paths:
    assert!(!vb.pp("a").contains_tensor("weight"));
    assert!(!vb.contains_tensor("weight"));
    assert!(!vb.pp("a").pp("b").pp("c").contains_tensor("weight"));
}

// ===========================================================================
// 2. Weight name mapping -- HfToNnMapper renames HF keys to nn convention
// ===========================================================================

#[test]
fn test_hf_mapper_identity_when_no_rules() {
    let mapper = HfToNnMapper::new();
    assert_eq!(mapper.map_name("encoder.weight"), "encoder.weight");
    assert_eq!(mapper.map_name(""), "");
}

#[test]
fn test_hf_mapper_prefix_rule_replaces_prefix() {
    let mapper = HfToNnMapper::new().with_prefix_rule("model.layers", "layers");
    // NN name "layers.0.weight" maps to HF "model.layers.0.weight".
    assert_eq!(mapper.map_name("layers.0.weight"), "model.layers.0.weight");
}

#[test]
fn test_hf_mapper_segment_rule_replaces_matching_segments() {
    let mapper = HfToNnMapper::new()
        .with_segment_rule("self_attn", "attn")
        .with_segment_rule("q_proj", "q");
    assert_eq!(
        mapper.map_name("layers.0.attn.q.weight"),
        "layers.0.self_attn.q_proj.weight"
    );
}

#[test]
fn test_hf_mapper_decoder_transformer_preset_full_chain() {
    let mapper = HfToNnMapper::decoder_transformer();
    // NN name -> HF checkpoint name
    assert_eq!(
        mapper.map_name("layers.0.attn.q.weight"),
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.attn.v.weight"),
        "model.layers.0.self_attn.v_proj.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.mlp.gate.weight"),
        "model.layers.0.mlp.gate_proj.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.ln1.weight"),
        "model.layers.0.input_layernorm.weight"
    );
}

#[test]
fn test_hf_mapper_exact_overrides_take_precedence() {
    let mut overrides = HashMap::new();
    overrides.insert(
        "embed.weight".to_string(),
        "model.embed_tokens.weight".to_string(),
    );
    let mapper = HfToNnMapper::new()
        .with_segment_rule("something_else", "embed")
        .with_exact_overrides(overrides);

    // Exact override wins:
    assert_eq!(mapper.map_name("embed.weight"), "model.embed_tokens.weight");
    // Non-override still goes through segment rules:
    assert_eq!(mapper.map_name("embed.bias"), "something_else.bias");
}

#[test]
fn test_hf_mapper_suffix_rule_appends_to_matching_bases() {
    let mapper = HfToNnMapper::new().with_suffix_rule("_proj", &["q", "k", "v", "o"]);
    assert_eq!(mapper.map_name("attn.q.weight"), "attn.q_proj.weight");
    assert_eq!(mapper.map_name("attn.k.weight"), "attn.k_proj.weight");
    assert_eq!(mapper.map_name("attn.v.weight"), "attn.v_proj.weight");
    assert_eq!(mapper.map_name("attn.o.weight"), "attn.o_proj.weight");
    // Non-matching segment unchanged:
    assert_eq!(mapper.map_name("attn.bias"), "attn.bias");
}

#[test]
fn test_hf_mapper_with_varbuilder_resolves_hf_key() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    let mapper = HfToNnMapper::decoder_transformer();
    let vb =
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu).with_weight_name_mapper(mapper);

    let t = vb
        .pp("layers")
        .pp("0")
        .pp("attn")
        .get(&[2, 2], "q.weight")
        .unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

// ===========================================================================
// 3. ZerosBackend -- produces zero tensors of requested shape/dtype
// ===========================================================================

#[test]
fn test_zeros_backend_f32_returns_all_zeros() {
    let backend = ZerosBackend;
    let t = backend
        .get(&[3, 4], "any_name", DType::F32, &Device::Cpu)
        .unwrap();
    assert_eq!(t.dims(), &[3, 4]);
    assert_eq!(t.dtype(), DType::F32);
    let data = t.to_flat_vec::<f32>().unwrap();
    assert_eq!(data.len(), 12);
    assert!(data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_zeros_backend_bf16_returns_correct_dtype() {
    let backend = ZerosBackend;
    let t = backend
        .get(&[2, 5], "w", DType::BF16, &Device::Cpu)
        .unwrap();
    assert_eq!(t.dims(), &[2, 5]);
    assert_eq!(t.dtype(), DType::BF16);
}

#[test]
fn test_zeros_backend_f16_returns_correct_dtype() {
    let backend = ZerosBackend;
    let t = backend.get(&[8], "w", DType::F16, &Device::Cpu).unwrap();
    assert_eq!(t.dims(), &[8]);
    assert_eq!(t.dtype(), DType::F16);
}

#[test]
fn test_zeros_backend_get_unchecked_returns_scalar() {
    let backend = ZerosBackend;
    let t = backend
        .get_unchecked("any", DType::F32, &Device::Cpu)
        .unwrap();
    assert_eq!(t.dims(), &[] as &[usize]);
}

#[test]
fn test_zeros_backend_contains_always_true() {
    let backend = ZerosBackend;
    assert!(backend.contains_tensor("anything"));
    assert!(backend.contains_tensor(""));
    assert!(backend.contains_tensor("deeply.nested.path"));
}

#[test]
fn test_zeros_backend_tensor_names_is_empty() {
    let backend = ZerosBackend;
    assert!(backend.tensor_names().is_empty());
}

#[test]
fn test_zeros_backend_high_rank_shape() {
    let backend = ZerosBackend;
    let t = backend
        .get(&[2, 3, 4, 5], "high_rank", DType::F32, &Device::Cpu)
        .unwrap();
    assert_eq!(t.dims(), &[2, 3, 4, 5]);
}

// ===========================================================================
// 4. TensorMapBackend -- loading from HashMap<String, DynTensor>
// ===========================================================================

#[test]
fn test_tensor_map_backend_retrieves_stored_tensors() {
    let mut map = HashMap::new();
    map.insert(
        "weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap(),
    );
    map.insert(
        "bias".to_string(),
        DynTensor::new(&[0.5], &[1], &Device::Cpu).unwrap(),
    );
    let backend = TensorMapBackend::new(map);

    let w = backend
        .get(&[3], "weight", DType::F32, &Device::Cpu)
        .unwrap();
    assert_eq!(w.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);

    let b = backend.get(&[1], "bias", DType::F32, &Device::Cpu).unwrap();
    assert_eq!(b.to_flat_vec::<f32>().unwrap(), vec![0.5]);
}

#[test]
fn test_tensor_map_backend_get_unchecked_skips_shape() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    let backend = TensorMapBackend::new(map);
    let t = backend
        .get_unchecked("w", DType::F32, &Device::Cpu)
        .unwrap();
    assert_eq!(t.dims(), &[2, 2]);
}

#[test]
fn test_tensor_map_backend_contains_existing_and_missing() {
    let mut map = HashMap::new();
    map.insert(
        "exists".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let backend = TensorMapBackend::new(map);
    assert!(backend.contains_tensor("exists"));
    assert!(!backend.contains_tensor("missing"));
}

#[test]
fn test_tensor_map_backend_tensor_names_returns_all_keys() {
    let mut map = HashMap::new();
    map.insert(
        "a.weight".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    map.insert(
        "b.bias".to_string(),
        DynTensor::new(&[2.0], &[1], &Device::Cpu).unwrap(),
    );
    let backend = TensorMapBackend::new(map);
    let mut names = backend.tensor_names();
    names.sort();
    assert_eq!(names, vec!["a.weight", "b.bias"]);
}

#[test]
fn test_tensor_map_backend_dtype_conversion_f32_to_bf16() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    let backend = TensorMapBackend::new(map);
    // Request BF16 from an F32 tensor -- should convert:
    let t = backend.get(&[2], "w", DType::BF16, &Device::Cpu).unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    assert_eq!(t.dims(), &[2]);
}

#[test]
fn test_tensor_map_backend_dtype_conversion_f32_to_f16() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[3.0, 4.0], &[2], &Device::Cpu).unwrap(),
    );
    let backend = TensorMapBackend::new(map);
    let t = backend.get(&[2], "w", DType::F16, &Device::Cpu).unwrap();
    assert_eq!(t.dtype(), DType::F16);
}

#[test]
fn test_tensor_map_backend_rejects_nan() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, f32::NAN], &[2], &Device::Cpu).unwrap(),
    );
    let backend = TensorMapBackend::new(map);
    let err = backend
        .get(&[2], "w", DType::F32, &Device::Cpu)
        .unwrap_err();
    match err {
        TensorError::NonFiniteData { name, count } => {
            assert_eq!(name, "w");
            assert_eq!(count, 1);
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

#[test]
fn test_tensor_map_backend_rejects_inf_via_get_unchecked() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[f32::INFINITY], &[1], &Device::Cpu).unwrap(),
    );
    let backend = TensorMapBackend::new(map);
    let err = backend
        .get_unchecked("w", DType::F32, &Device::Cpu)
        .unwrap_err();
    match err {
        TensorError::NonFiniteData { name, count } => {
            assert_eq!(name, "w");
            assert_eq!(count, 1);
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

// ===========================================================================
// 5. Safetensors round-trip -- save, reload via VarBuilder, verify equality
// ===========================================================================

#[test]
fn test_safetensors_bytes_roundtrip_then_varbuilder_access() {
    let mut original = HashMap::new();
    original.insert(
        "layer.0.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap(),
    );
    original.insert(
        "layer.0.bias".to_string(),
        DynTensor::new(&[0.5, -0.5, 0.0], &[3], &Device::Cpu).unwrap(),
    );
    original.insert(
        "layer.1.weight".to_string(),
        DynTensor::new(&[10.0, 20.0], &[2], &Device::Cpu).unwrap(),
    );

    let bytes = tensors_to_safetensors_bytes(&original).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    let vb = VarBuilder::from_tensors(loaded, DType::F32, &Device::Cpu);

    let w0 = vb.pp("layer").pp("0").get(&[3], "weight").unwrap();
    assert_eq!(w0.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);

    let b0 = vb.pp("layer").pp("0").get(&[3], "bias").unwrap();
    assert_eq!(b0.to_flat_vec::<f32>().unwrap(), vec![0.5, -0.5, 0.0]);

    let w1 = vb.pp("layer").pp("1").get(&[2], "weight").unwrap();
    assert_eq!(w1.to_flat_vec::<f32>().unwrap(), vec![10.0, 20.0]);
}

#[test]
fn test_safetensors_file_roundtrip_then_varbuilder() {
    let dir = std::env::temp_dir().join("nn_test_vb_weight_roundtrip");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("weights.safetensors");

    let mut tensors = HashMap::new();
    tensors.insert(
        "encoder.conv.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "encoder.conv.bias".to_string(),
        DynTensor::new(&[-1.0, 1.0], &[2], &Device::Cpu).unwrap(),
    );
    save_safetensors(&tensors, &path).unwrap();

    let loaded = load_safetensors(&path).unwrap();
    let vb = VarBuilder::from_tensors(loaded, DType::F32, &Device::Cpu);

    let w = vb.pp("encoder").pp("conv").get(&[2, 3], "weight").unwrap();
    assert_eq!(
        w.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    let b = vb.pp("encoder").pp("conv").get(&[2], "bias").unwrap();
    assert_eq!(b.to_flat_vec::<f32>().unwrap(), vec![-1.0, 1.0]);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_safetensors_roundtrip_with_hf_mapper() {
    // Save tensors with HF naming, load with NN names via mapper.
    let mut hf_tensors = HashMap::new();
    hf_tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    hf_tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        DynTensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2], &Device::Cpu).unwrap(),
    );

    let bytes = tensors_to_safetensors_bytes(&hf_tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();

    let mapper = HfToNnMapper::decoder_transformer();
    let vb =
        VarBuilder::from_tensors(loaded, DType::F32, &Device::Cpu).with_weight_name_mapper(mapper);

    let q = vb
        .pp("layers")
        .pp("0")
        .pp("attn")
        .get(&[2, 2], "q.weight")
        .unwrap();
    assert_eq!(q.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

    let k = vb
        .pp("layers")
        .pp("0")
        .pp("attn")
        .get(&[2, 2], "k.weight")
        .unwrap();
    assert_eq!(k.to_flat_vec::<f32>().unwrap(), vec![5.0, 6.0, 7.0, 8.0]);
}

// ===========================================================================
// 6. Error cases -- missing key, dtype mismatch, shape mismatch
// ===========================================================================

#[test]
fn test_error_missing_key_includes_full_path() {
    let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu);
    let err = vb
        .pp("model")
        .pp("encoder")
        .get(&[4], "weight")
        .unwrap_err();
    match err {
        TensorError::TensorNotFound { name } => {
            assert_eq!(name, "model.encoder.weight");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

#[test]
fn test_error_missing_key_get_unchecked() {
    let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu);
    let err = vb.pp("a").pp("b").get_unchecked("c").unwrap_err();
    match err {
        TensorError::TensorNotFound { name } => {
            assert_eq!(name, "a.b.c");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

#[test]
fn test_error_shape_mismatch_on_get() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let err = vb.get(&[2, 2], "w").unwrap_err();
    match err {
        TensorError::ShapeMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, vec![2, 2]);
            assert_eq!(actual, vec![3]);
        }
        other => panic!("expected ShapeMismatch, got: {other:?}"),
    }
}

#[test]
fn test_error_shape_mismatch_same_rank_different_dims() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let err = vb.get(&[3, 2], "w").unwrap_err();
    match err {
        TensorError::ShapeMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, vec![3, 2]);
            assert_eq!(actual, vec![2, 3]);
        }
        other => panic!("expected ShapeMismatch, got: {other:?}"),
    }
}

#[test]
fn test_error_nan_data_rejected_on_load() {
    let mut map = HashMap::new();
    map.insert(
        "bad".to_string(),
        DynTensor::new(&[f32::NAN, 1.0, f32::NAN], &[3], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let err = vb.get(&[3], "bad").unwrap_err();
    match err {
        TensorError::NonFiniteData { name, count } => {
            assert_eq!(name, "bad");
            assert_eq!(count, 2);
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

#[test]
fn test_error_inf_data_rejected_on_load() {
    let mut map = HashMap::new();
    map.insert(
        "inf".to_string(),
        DynTensor::new(&[f32::NEG_INFINITY], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let err = vb.get(&[1], "inf").unwrap_err();
    match err {
        TensorError::NonFiniteData { name, count } => {
            assert_eq!(name, "inf");
            assert_eq!(count, 1);
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

#[test]
fn test_error_missing_with_name_mapping() {
    // Name mapping transforms the key, so the error should reflect the mapped name.
    let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu)
        .with_name_mapping(|n| n.replace("nn_prefix", "hf_prefix"));
    let err = vb.pp("nn_prefix").get(&[1], "weight").unwrap_err();
    match err {
        TensorError::TensorNotFound { name } => {
            // The mapped name is what gets looked up:
            assert_eq!(name, "hf_prefix.weight");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

// ===========================================================================
// 7. WeightNameMapper -- verify mapper coverage utility
// ===========================================================================

#[test]
fn test_verify_coverage_all_mapped_returns_empty() {
    let mapper = HfToNnMapper::new().with_prefix_rule("model", "m");
    let checkpoint = vec!["model.w".to_string(), "model.b".to_string()];
    let nn = vec!["m.w".to_string(), "m.b".to_string()];

    let missing = verify_mapper_coverage(&nn, &checkpoint, &mapper);
    assert!(missing.is_empty());
}

#[test]
fn test_verify_coverage_reports_missing_names() {
    let mapper = HfToNnMapper::new();
    let checkpoint = vec!["a".to_string()];
    let nn = vec!["a".to_string(), "b".to_string(), "c".to_string()];

    let missing = verify_mapper_coverage(&nn, &checkpoint, &mapper);
    assert_eq!(missing.len(), 2);
    assert!(missing.contains(&"b".to_string()));
    assert!(missing.contains(&"c".to_string()));
}

#[test]
fn test_verify_coverage_empty_nn_names_returns_empty() {
    let mapper = HfToNnMapper::new();
    let checkpoint = vec!["some.weight".to_string()];
    let nn: Vec<String> = vec![];

    let missing = verify_mapper_coverage(&nn, &checkpoint, &mapper);
    assert!(missing.is_empty());
}

#[test]
fn test_verify_coverage_empty_checkpoint_names_all_missing() {
    let mapper = HfToNnMapper::new();
    let checkpoint: Vec<String> = vec![];
    let nn = vec!["a".to_string(), "b".to_string()];

    let missing = verify_mapper_coverage(&nn, &checkpoint, &mapper);
    assert_eq!(missing.len(), 2);
}

#[test]
fn test_verify_coverage_with_decoder_transformer_mapper() {
    let mapper = HfToNnMapper::decoder_transformer();
    let checkpoint = vec![
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        "model.layers.0.self_attn.v_proj.weight".to_string(),
        "model.layers.0.self_attn.o_proj.weight".to_string(),
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        "model.layers.0.mlp.up_proj.weight".to_string(),
        "model.layers.0.mlp.down_proj.weight".to_string(),
    ];
    let nn = vec![
        "layers.0.attn.q.weight".to_string(),
        "layers.0.attn.k.weight".to_string(),
        "layers.0.attn.v.weight".to_string(),
        "layers.0.attn.o.weight".to_string(),
        "layers.0.mlp.gate.weight".to_string(),
        "layers.0.mlp.up.weight".to_string(),
        "layers.0.mlp.down.weight".to_string(),
    ];

    let missing = verify_mapper_coverage(&nn, &checkpoint, &mapper);
    assert!(
        missing.is_empty(),
        "expected all mapped, but missing: {missing:?}"
    );
}

#[test]
fn test_verify_coverage_with_custom_mapper() {
    struct UpperMapper;
    impl WeightNameMapper for UpperMapper {
        fn map_name(&self, nn_name: &str) -> String {
            nn_name.to_uppercase()
        }
    }

    let checkpoint = vec!["WEIGHT".to_string(), "BIAS".to_string()];
    let nn = vec![
        "weight".to_string(),
        "bias".to_string(),
        "extra".to_string(),
    ];

    let missing = verify_mapper_coverage(&nn, &checkpoint, &UpperMapper);
    assert_eq!(missing, vec!["extra"]); // "EXTRA" not in checkpoint
}

#[test]
fn test_verify_coverage_with_exact_overrides() {
    let mut overrides = HashMap::new();
    overrides.insert(
        "special.weight".to_string(),
        "checkpoint.special_w".to_string(),
    );
    let mapper = HfToNnMapper::new().with_exact_overrides(overrides);

    let checkpoint = vec!["checkpoint.special_w".to_string()];
    let nn = vec!["special.weight".to_string()];

    let missing = verify_mapper_coverage(&nn, &checkpoint, &mapper);
    assert!(missing.is_empty());
}

// ===========================================================================
// 8. Mixed dtypes -- VarBuilder handles f32, f16, bf16 tensors
// ===========================================================================

#[test]
fn test_zeros_varbuilder_f32_produces_f32_tensor() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let t = vb.get(&[4], "w").unwrap();
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.dims(), &[4]);
}

#[test]
fn test_zeros_varbuilder_bf16_produces_bf16_tensor() {
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let t = vb.get(&[3, 2], "w").unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    assert_eq!(t.dims(), &[3, 2]);
}

#[test]
fn test_zeros_varbuilder_f16_produces_f16_tensor() {
    let vb = VarBuilder::zeros(DType::F16, &Device::Cpu);
    let t = vb.get(&[5], "w").unwrap();
    assert_eq!(t.dtype(), DType::F16);
    assert_eq!(t.dims(), &[5]);
}

#[test]
fn test_to_dtype_converts_varbuilder_output() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let vb_bf16 = vb.to_dtype(DType::BF16);
    let t = vb_bf16.get(&[3], "w").unwrap();
    assert_eq!(t.dtype(), DType::BF16);

    // Original still produces F32:
    let t_orig = vb.get(&[3], "w").unwrap();
    assert_eq!(t_orig.dtype(), DType::F32);
}

#[test]
fn test_tensor_map_backend_serves_f32_tensor_as_bf16_when_requested() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::BF16, &Device::Cpu);
    let t = vb.get(&[2, 2], "w").unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    assert_eq!(t.dims(), &[2, 2]);
}

#[test]
fn test_tensor_map_backend_serves_f32_tensor_as_f16_when_requested() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[5.0, 6.0], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F16, &Device::Cpu);
    let t = vb.get(&[2], "w").unwrap();
    assert_eq!(t.dtype(), DType::F16);
    assert_eq!(t.dims(), &[2]);
}

#[test]
fn test_mixed_dtype_to_dtype_chain() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let vb_f16 = vb.to_dtype(DType::F16);
    let vb_bf16 = vb.to_dtype(DType::BF16);

    assert_eq!(vb.dtype(), DType::F32);
    assert_eq!(vb_f16.dtype(), DType::F16);
    assert_eq!(vb_bf16.dtype(), DType::BF16);

    let t_f16 = vb_f16.get(&[2], "w").unwrap();
    assert_eq!(t_f16.dtype(), DType::F16);

    let t_bf16 = vb_bf16.get(&[2], "w").unwrap();
    assert_eq!(t_bf16.dtype(), DType::BF16);
}

#[test]
fn test_to_dtype_preserves_prefix_and_name_mapping() {
    let mut map = HashMap::new();
    map.insert(
        "checkpoint.weight".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
        .with_name_mapping(|n| n.replace("nn_name", "checkpoint"));

    let vb_bf16 = vb.pp("nn_name").to_dtype(DType::BF16);
    assert_eq!(vb_bf16.prefix(), "nn_name");
    assert_eq!(vb_bf16.dtype(), DType::BF16);
    assert!(vb_bf16.has_name_mapping());

    let t = vb_bf16.get(&[2], "weight").unwrap();
    assert_eq!(t.dtype(), DType::BF16);
}

#[test]
fn test_from_backend_custom_returns_requested_dtype() {
    struct ConstBackend;
    impl TensorBackend for ConstBackend {
        fn get(
            &self,
            dims: &[usize],
            _name: &str,
            dtype: DType,
            device: &Device,
        ) -> crate::Result<DynTensor> {
            DynTensor::zeros(dims, dtype, device)
        }
        fn get_unchecked(
            &self,
            _name: &str,
            dtype: DType,
            device: &Device,
        ) -> crate::Result<DynTensor> {
            DynTensor::zeros(&[1], dtype, device)
        }
        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    // F32:
    let vb_f32 = VarBuilder::from_backend(Arc::new(ConstBackend), DType::F32, Device::Cpu);
    assert_eq!(vb_f32.get(&[3], "w").unwrap().dtype(), DType::F32);

    // BF16:
    let vb_bf16 = VarBuilder::from_backend(Arc::new(ConstBackend), DType::BF16, Device::Cpu);
    assert_eq!(vb_bf16.get(&[3], "w").unwrap().dtype(), DType::BF16);

    // F16:
    let vb_f16 = VarBuilder::from_backend(Arc::new(ConstBackend), DType::F16, Device::Cpu);
    assert_eq!(vb_f16.get(&[3], "w").unwrap().dtype(), DType::F16);
}

#[test]
fn test_effective_weight_dtype_without_policy_matches_vb_dtype() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    assert_eq!(vb.effective_weight_dtype(), DType::F32);

    let vb_bf16 = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    assert_eq!(vb_bf16.effective_weight_dtype(), DType::BF16);
}

#[test]
fn test_effective_weight_dtype_with_precision_policy() {
    use crate::mixed_precision::MixedPrecisionPolicy;

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu).with_precision_policy(policy);

    // Apple Silicon default sets weight_dtype to BF16:
    assert_eq!(vb.effective_weight_dtype(), DType::BF16);

    // Propagates through pp():
    let child = vb.pp("encoder").pp("layer");
    assert_eq!(child.effective_weight_dtype(), DType::BF16);
}
