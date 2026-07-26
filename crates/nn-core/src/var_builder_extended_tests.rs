// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended VarBuilder tests covering safetensors roundtrip, verify_mapper_coverage,
//! TensorMapBackend direct usage, combined mapper+VarBuilder patterns, and additional
//! edge cases not exercised by existing test files.
//!
//! Part of #4186.

use std::collections::HashMap;
use std::sync::Arc;

use crate::dyn_tensor::DynTensor;
use crate::var_builder::{
    verify_mapper_coverage, HfToNnMapper, TensorBackend, TensorMapBackend, VarBuilder,
    WeightNameMapper, ZerosBackend,
};
use crate::{DType, Device, TensorError};

// ===========================================================================
// A. Safetensors roundtrip tests
// ===========================================================================

#[test]
fn test_safetensors_bytes_roundtrip_single_tensor() {
    use crate::dyn_tensor::{load_safetensors_from_bytes, tensors_to_safetensors_bytes};
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    assert!(loaded.contains_key("weight"));
    let t = &loaded["weight"];
    assert_eq!(t.dims(), &[2, 2]);
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_safetensors_bytes_roundtrip_multiple_tensors() {
    use crate::dyn_tensor::{load_safetensors_from_bytes, tensors_to_safetensors_bytes};
    let mut tensors = HashMap::new();
    tensors.insert(
        "encoder.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "encoder.bias".to_string(),
        DynTensor::new(&[0.1, 0.2, 0.3], &[3], &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "decoder.weight".to_string(),
        DynTensor::new(&[10.0, 20.0], &[2], &Device::Cpu).unwrap(),
    );
    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    assert_eq!(loaded.len(), 3);
    assert_eq!(
        loaded["encoder.weight"].to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        loaded["encoder.bias"].to_flat_vec::<f32>().unwrap(),
        vec![0.1, 0.2, 0.3]
    );
    assert_eq!(
        loaded["decoder.weight"].to_flat_vec::<f32>().unwrap(),
        vec![10.0, 20.0]
    );
}

#[test]
fn test_safetensors_bytes_roundtrip_preserves_shapes() {
    use crate::dyn_tensor::{load_safetensors_from_bytes, tensors_to_safetensors_bytes};
    let mut tensors = HashMap::new();
    tensors.insert(
        "scalar".to_string(),
        DynTensor::new(&[42.0], &[], &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "vec".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "mat".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "rank3".to_string(),
        DynTensor::zeros(&[2, 3, 4], DType::F32, &Device::Cpu).unwrap(),
    );
    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    // Scalar stored as [] internally gets roundtripped.
    // safetensors may store scalar as shape [], check element count.
    let scalar = &loaded["scalar"];
    assert_eq!(scalar.to_flat_vec::<f32>().unwrap(), vec![42.0]);
    assert_eq!(loaded["vec"].dims(), &[3]);
    assert_eq!(loaded["mat"].dims(), &[2, 3]);
    assert_eq!(loaded["rank3"].dims(), &[2, 3, 4]);
}

#[test]
fn test_safetensors_file_roundtrip() {
    use crate::dyn_tensor::{load_safetensors, save_safetensors};
    let dir = std::env::temp_dir().join("nn_test_safetensors_roundtrip");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_weights.safetensors");

    let mut tensors = HashMap::new();
    tensors.insert(
        "layer.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[4], &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "layer.bias".to_string(),
        DynTensor::new(&[0.5], &[1], &Device::Cpu).unwrap(),
    );
    save_safetensors(&tensors, &path).unwrap();
    assert!(path.exists());

    let loaded = load_safetensors(&path).unwrap();
    assert_eq!(loaded.len(), 2);
    assert_eq!(
        loaded["layer.weight"].to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        loaded["layer.bias"].to_flat_vec::<f32>().unwrap(),
        vec![0.5]
    );

    // Cleanup.
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_safetensors_empty_map_roundtrip() {
    use crate::dyn_tensor::{load_safetensors_from_bytes, tensors_to_safetensors_bytes};
    let tensors: HashMap<String, DynTensor> = HashMap::new();
    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    assert!(loaded.is_empty());
}

#[test]
fn test_safetensors_roundtrip_into_varbuilder() {
    use crate::dyn_tensor::{load_safetensors_from_bytes, tensors_to_safetensors_bytes};
    let mut tensors = HashMap::new();
    tensors.insert(
        "model.encoder.weight".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "model.encoder.bias".to_string(),
        DynTensor::new(&[0.5, -0.5], &[2], &Device::Cpu).unwrap(),
    );
    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();

    // Feed into VarBuilder for hierarchical access.
    let vb = VarBuilder::from_tensors(loaded, DType::F32, &Device::Cpu);
    let w = vb.pp("model").pp("encoder").get(&[2], "weight").unwrap();
    let b = vb.pp("model").pp("encoder").get(&[2], "bias").unwrap();
    assert_eq!(w.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(b.to_flat_vec::<f32>().unwrap(), vec![0.5, -0.5]);
}

#[test]
fn test_safetensors_large_tensor_roundtrip() {
    use crate::dyn_tensor::{load_safetensors_from_bytes, tensors_to_safetensors_bytes};
    let data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.01).collect();
    let mut tensors = HashMap::new();
    tensors.insert(
        "large".to_string(),
        DynTensor::new(&data, &[32, 32], &Device::Cpu).unwrap(),
    );
    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    let t = &loaded["large"];
    assert_eq!(t.dims(), &[32, 32]);
    let loaded_data = t.to_flat_vec::<f32>().unwrap();
    assert_eq!(loaded_data.len(), 1024);
    // Check first and last values.
    assert!((loaded_data[0] - 0.0).abs() < 1e-6);
    assert!((loaded_data[1023] - 10.23).abs() < 1e-4);
}

#[test]
fn test_safetensors_negative_values_roundtrip() {
    use crate::dyn_tensor::{load_safetensors_from_bytes, tensors_to_safetensors_bytes};
    let mut tensors = HashMap::new();
    tensors.insert(
        "neg".to_string(),
        DynTensor::new(
            &[-1.0, -0.5, 0.0, 0.5, 1.0, f32::MIN_POSITIVE],
            &[6],
            &Device::Cpu,
        )
        .unwrap(),
    );
    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    let data = loaded["neg"].to_flat_vec::<f32>().unwrap();
    assert_eq!(data[0], -1.0);
    assert_eq!(data[1], -0.5);
    assert_eq!(data[2], 0.0);
    assert_eq!(data[3], 0.5);
    assert_eq!(data[4], 1.0);
    assert_eq!(data[5], f32::MIN_POSITIVE);
}

// ===========================================================================
// B. verify_mapper_coverage tests
// ===========================================================================

#[test]
fn test_verify_mapper_coverage_all_covered() {
    let mapper = HfToNnMapper::new().with_prefix_rule("model", "m");

    let checkpoint_names = vec!["model.weight".to_string(), "model.bias".to_string()];
    let nn_names = vec!["m.weight".to_string(), "m.bias".to_string()];

    let missing = verify_mapper_coverage(&nn_names, &checkpoint_names, &mapper);
    assert!(missing.is_empty(), "all names should be covered");
}

#[test]
fn test_verify_mapper_coverage_some_missing() {
    let mapper = HfToNnMapper::new().with_prefix_rule("model", "m");

    let checkpoint_names = vec!["model.weight".to_string(), "model.bias".to_string()];
    let nn_names = vec![
        "m.weight".to_string(),
        "m.bias".to_string(),
        "m.extra".to_string(),
    ];

    let missing = verify_mapper_coverage(&nn_names, &checkpoint_names, &mapper);
    assert_eq!(missing, vec!["m.extra"]);
}

#[test]
fn test_verify_mapper_coverage_empty_nn_names() {
    let mapper = HfToNnMapper::new();
    let checkpoint_names = vec!["model.weight".to_string()];
    let nn_names: Vec<String> = vec![];

    let missing = verify_mapper_coverage(&nn_names, &checkpoint_names, &mapper);
    assert!(missing.is_empty());
}

#[test]
fn test_verify_mapper_coverage_empty_checkpoint_names() {
    let mapper = HfToNnMapper::new();
    let checkpoint_names: Vec<String> = vec![];
    let nn_names = vec!["weight".to_string()];

    let missing = verify_mapper_coverage(&nn_names, &checkpoint_names, &mapper);
    assert_eq!(missing, vec!["weight"]);
}

#[test]
fn test_verify_mapper_coverage_identity_mapper() {
    let mapper = HfToNnMapper::new(); // identity
    let checkpoint_names = vec!["a.weight".to_string(), "b.weight".to_string()];
    let nn_names = vec!["a.weight".to_string(), "b.weight".to_string()];

    let missing = verify_mapper_coverage(&nn_names, &checkpoint_names, &mapper);
    assert!(missing.is_empty());
}

#[test]
fn test_verify_mapper_coverage_complex_mapper() {
    let mapper = HfToNnMapper::decoder_transformer();
    let checkpoint_names = vec![
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        "model.layers.0.mlp.gate_proj.weight".to_string(),
    ];
    let nn_names = vec![
        "layers.0.attn.q.weight".to_string(),
        "layers.0.attn.k.weight".to_string(),
        "layers.0.mlp.gate.weight".to_string(),
    ];

    let missing = verify_mapper_coverage(&nn_names, &checkpoint_names, &mapper);
    assert!(
        missing.is_empty(),
        "decoder_transformer mapper should cover all: {missing:?}"
    );
}

#[test]
fn test_verify_mapper_coverage_with_exact_overrides() {
    let mut overrides = HashMap::new();
    overrides.insert(
        "special.weight".to_string(),
        "checkpoint.special_w".to_string(),
    );

    let mapper = HfToNnMapper::new().with_exact_overrides(overrides);

    let checkpoint_names = vec!["checkpoint.special_w".to_string()];
    let nn_names = vec!["special.weight".to_string()];

    let missing = verify_mapper_coverage(&nn_names, &checkpoint_names, &mapper);
    assert!(missing.is_empty());
}

#[test]
fn test_verify_mapper_coverage_partial_coverage_reports_all_missing() {
    let mapper = HfToNnMapper::new();
    let checkpoint_names = vec!["a".to_string()];
    let nn_names = vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
    ];

    let missing = verify_mapper_coverage(&nn_names, &checkpoint_names, &mapper);
    assert_eq!(missing.len(), 3);
    assert!(missing.contains(&"b".to_string()));
    assert!(missing.contains(&"c".to_string()));
    assert!(missing.contains(&"d".to_string()));
}

// ===========================================================================
// C. TensorMapBackend direct usage
// ===========================================================================

#[test]
fn test_tensor_map_backend_get_validates_shape() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap(),
    );
    let backend = TensorMapBackend::new(map);
    let t = backend.get(&[3], "w", DType::F32, &Device::Cpu).unwrap();
    assert_eq!(t.dims(), &[3]);

    let err = backend
        .get(&[2], "w", DType::F32, &Device::Cpu)
        .unwrap_err();
    match err {
        TensorError::ShapeMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, vec![2]);
            assert_eq!(actual, vec![3]);
        }
        other => panic!("expected ShapeMismatch, got: {other:?}"),
    }
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
fn test_tensor_map_backend_contains_tensor() {
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
fn test_tensor_map_backend_tensor_names() {
    let mut map = HashMap::new();
    map.insert(
        "a".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    map.insert(
        "b".to_string(),
        DynTensor::new(&[2.0], &[1], &Device::Cpu).unwrap(),
    );
    let backend = TensorMapBackend::new(map);
    let mut names = backend.tensor_names();
    names.sort();
    assert_eq!(names, vec!["a", "b"]);
}

#[test]
fn test_tensor_map_backend_not_found_error() {
    let backend = TensorMapBackend::new(HashMap::new());
    let err = backend
        .get(&[1], "missing", DType::F32, &Device::Cpu)
        .unwrap_err();
    match err {
        TensorError::TensorNotFound { name } => {
            assert_eq!(name, "missing");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

#[test]
fn test_tensor_map_backend_rejects_nan_values() {
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
// D. ZerosBackend direct usage
// ===========================================================================

#[test]
fn test_zeros_backend_get_returns_zeros_with_correct_shape() {
    let backend = ZerosBackend;
    let t = backend
        .get(&[3, 4], "any_name", DType::F32, &Device::Cpu)
        .unwrap();
    assert_eq!(t.dims(), &[3, 4]);
    assert_eq!(t.dtype(), DType::F32);
    let data = t.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| v == 0.0));
    assert_eq!(data.len(), 12);
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
    assert!(backend.contains_tensor("deeply.nested.name"));
}

#[test]
fn test_zeros_backend_tensor_names_empty() {
    let backend = ZerosBackend;
    assert!(backend.tensor_names().is_empty());
}

#[test]
fn test_zeros_backend_bf16_dtype() {
    let backend = ZerosBackend;
    let t = backend.get(&[4], "w", DType::BF16, &Device::Cpu).unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    assert_eq!(t.dims(), &[4]);
}

// ===========================================================================
// E. Combined VarBuilder + safetensors + mapper patterns
// ===========================================================================

#[test]
fn test_safetensors_roundtrip_with_name_mapper() {
    use crate::dyn_tensor::{load_safetensors_from_bytes, tensors_to_safetensors_bytes};

    // Simulate: save HF-style names, load with NN mapper.
    let mut hf_tensors = HashMap::new();
    hf_tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    let bytes = tensors_to_safetensors_bytes(&hf_tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();

    let mapper = HfToNnMapper::decoder_transformer();
    let vb =
        VarBuilder::from_tensors(loaded, DType::F32, &Device::Cpu).with_weight_name_mapper(mapper);

    let t = vb
        .pp("layers")
        .pp("0")
        .pp("attn")
        .get(&[2, 2], "q.weight")
        .unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_varbuilder_with_rename_map_and_safetensors() {
    use crate::dyn_tensor::{load_safetensors_from_bytes, tensors_to_safetensors_bytes};

    let mut tensors = HashMap::new();
    tensors.insert(
        "checkpoint.enc.w".to_string(),
        DynTensor::new(&[7.0, 8.0], &[2], &Device::Cpu).unwrap(),
    );
    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();

    let rename = HashMap::from([("encoder.weight".to_string(), "checkpoint.enc.w".to_string())]);
    let vb = VarBuilder::from_tensors(loaded, DType::F32, &Device::Cpu).with_rename_map(rename);

    let t = vb.pp("encoder").get(&[2], "weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![7.0, 8.0]);
}

// ===========================================================================
// F. WeightNameMapper trait custom implementation
// ===========================================================================

#[test]
fn test_custom_weight_name_mapper_implementation() {
    /// Mapper that reverses dot-separated segments.
    struct ReverseMapper;
    impl WeightNameMapper for ReverseMapper {
        fn map_name(&self, nn_name: &str) -> String {
            let segments: Vec<&str> = nn_name.split('.').collect();
            let reversed: Vec<&str> = segments.into_iter().rev().collect();
            reversed.join(".")
        }
        fn description(&self) -> &'static str {
            "ReverseMapper"
        }
    }

    let mapper = ReverseMapper;
    assert_eq!(mapper.map_name("a.b.c"), "c.b.a");
    assert_eq!(mapper.map_name("weight"), "weight");
    assert_eq!(mapper.description(), "ReverseMapper");
}

#[test]
fn test_custom_weight_name_mapper_with_varbuilder() {
    /// Mapper that adds "checkpoint." prefix.
    struct PrefixMapper;
    impl WeightNameMapper for PrefixMapper {
        fn map_name(&self, nn_name: &str) -> String {
            format!("checkpoint.{nn_name}")
        }
    }

    let mut tensors = HashMap::new();
    tensors.insert(
        "checkpoint.encoder.weight".to_string(),
        DynTensor::new(&[3.0, 4.0], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu)
        .with_weight_name_mapper(PrefixMapper);

    let t = vb.pp("encoder").get(&[2], "weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![3.0, 4.0]);
}

#[test]
fn test_verify_mapper_coverage_with_custom_mapper() {
    struct UpperMapper;
    impl WeightNameMapper for UpperMapper {
        fn map_name(&self, nn_name: &str) -> String {
            nn_name.to_uppercase()
        }
    }

    let checkpoint_names = vec!["WEIGHT".to_string(), "BIAS".to_string()];
    let nn_names = vec![
        "weight".to_string(),
        "bias".to_string(),
        "extra".to_string(),
    ];

    let missing = verify_mapper_coverage(&nn_names, &checkpoint_names, &UpperMapper);
    assert_eq!(missing, vec!["extra"]); // "EXTRA" not in checkpoint
}

// ===========================================================================
// G. HfToNnMapper edge cases
// ===========================================================================

#[test]
fn test_hf_mapper_default_trait() {
    let mapper = HfToNnMapper::default();
    assert_eq!(mapper.map_name("x.y.z"), "x.y.z"); // identity
    assert_eq!(mapper.description(), "HfToNnMapper");
}

#[test]
fn test_hf_mapper_with_description() {
    let mapper = HfToNnMapper::new().with_description("custom-desc");
    assert_eq!(mapper.description(), "custom-desc");
}

#[test]
fn test_hf_mapper_prefix_rule_strips_empty_hf_prefix() {
    // NN has "model.encoder.weight", HF has just "encoder.weight" (strip prefix).
    let mapper = HfToNnMapper::new().with_prefix_rule("", "model");
    assert_eq!(mapper.map_name("model.encoder.weight"), "encoder.weight");
    // Just "model" with no remainder.
    assert_eq!(mapper.map_name("model"), "");
}

#[test]
fn test_hf_mapper_multiple_segment_rules_first_wins_per_segment() {
    let mapper = HfToNnMapper::new()
        .with_segment_rule("first_match", "seg")
        .with_segment_rule("second_match", "seg");
    // First matching segment rule wins.
    assert_eq!(mapper.map_name("a.seg.b"), "a.first_match.b");
}

#[test]
fn test_hf_mapper_segment_rule_does_not_affect_non_matching() {
    let mapper = HfToNnMapper::new().with_segment_rule("replaced", "target");
    assert_eq!(mapper.map_name("a.b.c"), "a.b.c"); // no "target" segment
}

#[test]
fn test_hf_mapper_suffix_rule_multiple_bases() {
    let mapper = HfToNnMapper::new().with_suffix_rule("_proj", &["q", "k", "v"]);
    assert_eq!(mapper.map_name("a.q.b"), "a.q_proj.b");
    assert_eq!(mapper.map_name("a.k.b"), "a.k_proj.b");
    assert_eq!(mapper.map_name("a.v.b"), "a.v_proj.b");
    assert_eq!(mapper.map_name("a.o.b"), "a.o.b"); // not in bases
}

#[test]
fn test_hf_mapper_exact_overrides_bypass_rules() {
    let mut overrides = HashMap::new();
    overrides.insert(
        "bypass.weight".to_string(),
        "exact_checkpoint_key".to_string(),
    );

    let mapper = HfToNnMapper::new()
        .with_segment_rule("something_else", "bypass") // would normally apply
        .with_exact_overrides(overrides);

    assert_eq!(mapper.map_name("bypass.weight"), "exact_checkpoint_key");
    // Non-override names still go through rules.
    assert_eq!(mapper.map_name("bypass.bias"), "something_else.bias");
}

#[test]
fn test_hf_mapper_empty_name() {
    let mapper = HfToNnMapper::new().with_segment_rule("x", "y");
    assert_eq!(mapper.map_name(""), "");
}

// ===========================================================================
// H. VarBuilder with_prefix_mapping edge cases
// ===========================================================================

#[test]
fn test_prefix_mapping_multiple_rules_first_match_wins() {
    let mut map = HashMap::new();
    map.insert(
        "first_target.weight".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
        .with_prefix_mapping(&[("src", "first_target"), ("src", "second_target")]);
    let t = vb.get(&[1], "src.weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0]);
}

#[test]
fn test_prefix_mapping_no_match_passes_through() {
    let mut map = HashMap::new();
    map.insert(
        "original.weight".to_string(),
        DynTensor::new(&[5.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
        .with_prefix_mapping(&[("nonexistent_prefix", "replacement")]);
    let t = vb.get(&[1], "original.weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![5.0]);
}

// ===========================================================================
// I. VarBuilder combined dtype + precision + mapper
// ===========================================================================

#[test]
fn test_varbuilder_to_dtype_with_name_mapping() {
    let mut map = HashMap::new();
    map.insert(
        "mapped.weight".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
        .with_name_mapping(|n| n.replace("original", "mapped"));
    let vb_bf16 = vb.to_dtype(DType::BF16);

    assert_eq!(vb_bf16.dtype(), DType::BF16);
    assert!(vb_bf16.has_name_mapping());
    let t = vb_bf16.get(&[2], "original.weight").unwrap();
    assert_eq!(t.dtype(), DType::BF16);
}

#[test]
fn test_varbuilder_precision_policy_with_name_mapping() {
    use crate::mixed_precision::MixedPrecisionPolicy;

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu)
        .with_name_mapping(ToString::to_string)
        .with_precision_policy(policy.clone());

    assert!(vb.has_name_mapping());
    assert_eq!(vb.precision_policy(), Some(&policy));
    assert_eq!(vb.effective_weight_dtype(), DType::BF16);

    // These properties propagate through pp().
    let child = vb.pp("encoder");
    assert!(child.has_name_mapping());
    assert_eq!(child.precision_policy(), Some(&policy));
}

// ===========================================================================
// J. VarBuilder from_backend with custom backends
// ===========================================================================

#[test]
fn test_from_backend_with_custom_tensor_names() {
    struct NamedBackend;
    impl TensorBackend for NamedBackend {
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
        fn contains_tensor(&self, name: &str) -> bool {
            name == "known_weight"
        }
        fn tensor_names(&self) -> Vec<String> {
            vec!["known_weight".to_string()]
        }
    }

    let vb = VarBuilder::from_backend(Arc::new(NamedBackend), DType::F32, Device::Cpu);
    assert!(vb.contains_tensor("known_weight"));
    assert!(!vb.contains_tensor("unknown"));
    assert_eq!(vb.tensor_names(), vec!["known_weight"]);
}

#[test]
fn test_from_backend_error_propagation() {
    struct FailBackend;
    impl TensorBackend for FailBackend {
        fn get(
            &self,
            _dims: &[usize],
            name: &str,
            _dtype: DType,
            _device: &Device,
        ) -> crate::Result<DynTensor> {
            Err(TensorError::TensorNotFound {
                name: name.to_string(),
            })
        }
        fn get_unchecked(
            &self,
            name: &str,
            _dtype: DType,
            _device: &Device,
        ) -> crate::Result<DynTensor> {
            Err(TensorError::TensorNotFound {
                name: name.to_string(),
            })
        }
        fn contains_tensor(&self, _name: &str) -> bool {
            false
        }
    }

    let vb = VarBuilder::from_backend(Arc::new(FailBackend), DType::F32, Device::Cpu);
    let err = vb.pp("model").get(&[1], "weight").unwrap_err();
    match err {
        TensorError::TensorNotFound { name } => {
            assert_eq!(name, "model.weight");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

// ===========================================================================
// K. VarBuilder thread-safety and clone
// ===========================================================================

#[test]
fn test_varbuilder_clone_independence() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu).pp("root");
    let vb2 = vb.clone();
    let vb3 = vb2.pp("child");

    // Original is unaffected by derived clones.
    assert_eq!(vb.prefix(), "root");
    assert_eq!(vb2.prefix(), "root");
    assert_eq!(vb3.prefix(), "root.child");
}

#[test]
fn test_varbuilder_send_to_thread() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);

    let handle = std::thread::spawn(move || {
        let t = vb.get(&[2], "w").unwrap();
        t.to_flat_vec::<f32>().unwrap()
    });
    let result = handle.join().unwrap();
    assert_eq!(result, vec![1.0, 2.0]);
}

#[test]
fn test_varbuilder_shared_across_threads() {
    let mut map = HashMap::new();
    map.insert(
        "shared.w".to_string(),
        DynTensor::new(&[10.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = Arc::new(VarBuilder::from_tensors(map, DType::F32, &Device::Cpu));

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let vb_clone = Arc::clone(&vb);
            std::thread::spawn(move || {
                let t = vb_clone.pp("shared").get(&[1], "w").unwrap();
                t.to_flat_vec::<f32>().unwrap()
            })
        })
        .collect();

    for handle in handles {
        let result = handle.join().unwrap();
        assert_eq!(result, vec![10.0]);
    }
}
