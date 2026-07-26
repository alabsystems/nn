#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for VarBuilder (D5 of #914, #915).

use std::collections::HashMap;

use crate::dyn_tensor::DynTensor;
use crate::var_builder::VarBuilder;
use crate::{DType, Device, TensorError};

// -- pp() tests ---------------------------------------------------------------

#[test]
fn test_pp_builds_dot_separated_path() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let enc = vb.pp("encoder");
    assert_eq!(enc.prefix(), "encoder");
}

#[test]
fn test_pp_nested_chains() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let deep = vb.pp("model").pp("encoder").pp("layer0");
    assert_eq!(deep.prefix(), "model.encoder.layer0");
}

#[test]
fn test_pp_empty_prefix() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    assert_eq!(vb.prefix(), "");
}

#[test]
fn test_pp_empty_string_is_skipped() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let scoped = vb.pp("encoder").pp("").pp("layer0");
    // Empty prefix should be skipped — no leading/trailing/double dots.
    assert_eq!(scoped.prefix(), "encoder.layer0");
}

// -- zeros backend tests ------------------------------------------------------

#[test]
fn test_zeros_get_returns_zero_tensor() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let t = vb.get(&[3, 4], "weight").unwrap();
    assert_eq!(t.dims(), &[3, 4]);
    assert_eq!(t.dtype(), DType::F32);
    let data = t.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_zeros_contains_always_true() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    assert!(vb.contains_tensor("anything"));
    assert!(vb.pp("deep").pp("nested").contains_tensor("weight"));
}

#[test]
fn test_zeros_get_unchecked_returns_scalar() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let t = vb.get_unchecked("bias").unwrap();
    // ZerosBackend::get_unchecked returns a 0-D scalar (shape []),
    // not a 1-D [1] tensor. Fixed in 8622bbab.
    assert_eq!(t.dims(), &[] as &[usize]);
}

// -- tensor map backend tests -------------------------------------------------

fn sample_tensors() -> HashMap<String, DynTensor> {
    let mut map = HashMap::new();
    map.insert(
        "encoder.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu).unwrap(),
    );
    map.insert(
        "encoder.bias".to_string(),
        DynTensor::new(&[0.1, 0.2], &[2], &Device::Cpu).unwrap(),
    );
    map.insert(
        "decoder.weight".to_string(),
        DynTensor::new(&[7.0, 8.0, 9.0], &[3], &Device::Cpu).unwrap(),
    );
    map
}

#[test]
fn test_tensor_map_get_returns_correct_tensor() {
    let vb = VarBuilder::from_tensors(sample_tensors(), DType::F32, &Device::Cpu);
    let enc = vb.pp("encoder");
    let w = enc.get(&[2, 3], "weight").unwrap();
    assert_eq!(w.dims(), &[2, 3]);
    let data = w.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_tensor_map_shape_mismatch() {
    let vb = VarBuilder::from_tensors(sample_tensors(), DType::F32, &Device::Cpu);
    let enc = vb.pp("encoder");
    let err = enc.get(&[3, 2], "weight").unwrap_err(); // wrong shape
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
fn test_tensor_map_not_found() {
    let vb = VarBuilder::from_tensors(sample_tensors(), DType::F32, &Device::Cpu);
    let err = vb.pp("encoder").get(&[1], "nonexistent").unwrap_err();
    match err {
        TensorError::TensorNotFound { name } => {
            assert_eq!(name, "encoder.nonexistent");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

#[test]
fn test_tensor_map_get_unchecked_skips_shape() {
    let vb = VarBuilder::from_tensors(sample_tensors(), DType::F32, &Device::Cpu);
    let t = vb.pp("encoder").get_unchecked("weight").unwrap();
    // Shape isn't validated, so we get the stored shape.
    assert_eq!(t.dims(), &[2, 3]);
}

#[test]
fn test_tensor_map_contains_with_prefix() {
    let vb = VarBuilder::from_tensors(sample_tensors(), DType::F32, &Device::Cpu);
    let enc = vb.pp("encoder");
    assert!(enc.contains_tensor("weight"));
    assert!(enc.contains_tensor("bias"));
    assert!(!enc.contains_tensor("nonexistent"));
}

#[test]
fn test_tensor_map_get_no_prefix() {
    let vb = VarBuilder::from_tensors(sample_tensors(), DType::F32, &Device::Cpu);
    // Direct access with full key (no pp).
    let t = vb.get(&[3], "decoder.weight").unwrap();
    assert_eq!(t.dims(), &[3]);
    let data = t.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![7.0, 8.0, 9.0]);
}

// -- dtype/device accessors ---------------------------------------------------

#[test]
fn test_dtype_and_device_accessors() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    assert_eq!(vb.dtype(), DType::F32);
    assert_eq!(*vb.device(), Device::Cpu);
}

#[test]
fn test_to_dtype_changes_dtype() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let vb2 = vb.to_dtype(DType::BF16);
    assert_eq!(vb2.dtype(), DType::BF16);
    assert_eq!(vb.dtype(), DType::F32); // original unchanged
}

#[test]
fn test_to_device_changes_device() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let vb2 = vb.to_device(Device::metal());
    assert_eq!(*vb2.device(), Device::metal());
    assert_eq!(*vb.device(), Device::Cpu); // original unchanged
}

// -- sharing tests ------------------------------------------------------------

#[test]
fn test_pp_shares_backend_arc() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let vb2 = vb.pp("a");
    let vb3 = vb2.pp("b");
    // All three can retrieve tensors — sharing the same backend.
    assert!(vb.get(&[1], "x").is_ok());
    assert!(vb2.get(&[1], "x").is_ok());
    assert!(vb3.get(&[1], "x").is_ok());
}

// -- debug test ---------------------------------------------------------------

#[test]
fn test_debug_format() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu).pp("encoder");
    let debug = format!("{vb:?}");
    assert!(debug.contains("encoder"), "debug should show prefix");
    assert!(debug.contains("VarBuilder"), "debug should show type name");
}

// -- weight finiteness validation tests (#943) --------------------------------

#[test]
fn test_tensor_map_rejects_nan_weight_via_get() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, f32::NAN, 3.0], &[3], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let err = vb.get(&[3], "w").unwrap_err();
    match err {
        TensorError::NonFiniteData { name, count } => {
            assert_eq!(name, "w");
            assert_eq!(count, 1);
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

#[test]
fn test_tensor_map_rejects_inf_weight_via_get() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[f32::INFINITY, f32::NEG_INFINITY, 1.0], &[3], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let err = vb.get(&[3], "w").unwrap_err();
    match err {
        TensorError::NonFiniteData { name, count } => {
            assert_eq!(name, "w");
            assert_eq!(count, 2);
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

#[test]
fn test_tensor_map_rejects_nan_weight_via_get_unchecked() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[f32::NAN], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let err = vb.get_unchecked("w").unwrap_err();
    match err {
        TensorError::NonFiniteData { name, count } => {
            assert_eq!(name, "w");
            assert_eq!(count, 1);
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

#[test]
fn test_tensor_map_finite_weights_pass_through() {
    // Regression test: finite tensors must not be rejected.
    let vb = VarBuilder::from_tensors(sample_tensors(), DType::F32, &Device::Cpu);
    let w = vb.pp("encoder").get(&[2, 3], "weight").unwrap();
    assert_eq!(w.dims(), &[2, 3]);
    let b = vb.pp("encoder").get_unchecked("bias").unwrap();
    assert_eq!(b.dims(), &[2]);
}

// -- precision policy tests ---------------------------------------------------

#[test]
fn test_with_precision_policy_sets_policy() {
    use crate::mixed_precision::MixedPrecisionPolicy;
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu).with_precision_policy(policy.clone());
    assert_eq!(vb.precision_policy(), Some(&policy));
}

#[test]
fn test_effective_weight_dtype_with_policy() {
    use crate::mixed_precision::MixedPrecisionPolicy;
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu).with_precision_policy(policy);
    // Policy weight_dtype is BF16
    assert_eq!(vb.effective_weight_dtype(), DType::BF16);
}

#[test]
fn test_effective_weight_dtype_without_policy() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    // No policy: falls back to VarBuilder's dtype
    assert_eq!(vb.effective_weight_dtype(), DType::F32);
    assert!(vb.precision_policy().is_none());
}

#[test]
fn test_precision_policy_propagates_through_pp() {
    use crate::mixed_precision::MixedPrecisionPolicy;
    let policy = MixedPrecisionPolicy::cuda_bf16();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu).with_precision_policy(policy.clone());
    let child = vb.pp("encoder").pp("layer0");
    // Policy should propagate through pp()
    assert_eq!(child.precision_policy(), Some(&policy));
    assert_eq!(child.effective_weight_dtype(), DType::BF16);
}

// -- name mapping tests (#2422) -----------------------------------------------

#[test]
fn test_with_name_mapping_transforms_keys() {
    let mut map = HashMap::new();
    map.insert(
        "vision.encoder.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
        .with_name_mapping(|name| name.replace("model.encoder", "vision.encoder"));
    let t = vb.pp("model").pp("encoder").get(&[3], "weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_with_prefix_mapping() {
    let mut map = HashMap::new();
    map.insert(
        "vision_model.encoder.layers.0.weight".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
        .with_prefix_mapping(&[("encoder.layer.", "vision_model.encoder.layers.")]);
    let t = vb.pp("encoder").pp("layer.0").get(&[2], "weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn test_name_mapping_propagates_through_pp() {
    let mut map = HashMap::new();
    map.insert(
        "mapped.a.b.weight".to_string(),
        DynTensor::new(&[7.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
        .with_name_mapping(|name| name.replace("orig", "mapped"));
    let child = vb.pp("orig").pp("a").pp("b");
    assert!(child.has_name_mapping());
    let t = child.get(&[1], "weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![7.0]);
}

#[test]
fn test_name_mapping_contains_tensor() {
    let mut map = HashMap::new();
    map.insert(
        "real.weight".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
        .with_name_mapping(|name| name.replace("fake", "real"));
    assert!(vb.pp("fake").contains_tensor("weight"));
    assert!(!vb.pp("fake").contains_tensor("bias"));
}

#[test]
fn test_name_mapping_get_unchecked() {
    let mut map = HashMap::new();
    map.insert(
        "actual.w".to_string(),
        DynTensor::new(&[5.0, 6.0], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
        .with_name_mapping(|name| name.replace("alias", "actual"));
    let t = vb.pp("alias").get_unchecked("w").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![5.0, 6.0]);
}

#[test]
fn test_no_name_mapping_by_default() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    assert!(!vb.has_name_mapping());
}

#[test]
fn test_prefix_mapping_first_match_wins() {
    let mut map = HashMap::new();
    map.insert(
        "first.weight".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
        .with_prefix_mapping(&[("a", "first"), ("a", "second")]);
    // First matching prefix wins
    let t = vb.get(&[1], "a.weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0]);
}

// -- rename map tests (#2422) -------------------------------------------------

#[test]
fn test_with_rename_map_remaps_exact_keys() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "model.vision_model.encoder.layers.0.self_attn.q_proj.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    let rename = HashMap::from([(
        "encoder.layer.0.q.weight".to_string(),
        "model.vision_model.encoder.layers.0.self_attn.q_proj.weight".to_string(),
    )]);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu).with_rename_map(rename);
    // NN model code requests "encoder.layer.0.q.weight" → backend looks up the HF name.
    let t = vb.get(&[2, 2], "encoder.layer.0.q.weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_with_rename_map_passthrough_unmapped_keys() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "decoder.weight".to_string(),
        DynTensor::new(&[5.0, 6.0], &[2], &Device::Cpu).unwrap(),
    );
    let rename = HashMap::from([(
        "encoder.weight".to_string(),
        "model.encoder.weight".to_string(),
    )]);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu).with_rename_map(rename);
    // "decoder.weight" is not in the rename map → passes through unchanged.
    let t = vb.get(&[2], "decoder.weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![5.0, 6.0]);
}

#[test]
fn test_with_rename_map_with_pp_prefix() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "hf.layers.0.attn.q.weight".to_string(),
        DynTensor::new(&[9.0], &[1], &Device::Cpu).unwrap(),
    );
    let rename = HashMap::from([(
        "model.layers.0.attn.q.weight".to_string(),
        "hf.layers.0.attn.q.weight".to_string(),
    )]);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu).with_rename_map(rename);
    // pp("model").pp("layers").pp("0").pp("attn").pp("q") + "weight"
    // resolves to "model.layers.0.attn.q.weight" → remapped to HF name.
    let t = vb
        .pp("model")
        .pp("layers")
        .pp("0")
        .pp("attn")
        .pp("q")
        .get(&[1], "weight")
        .unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![9.0]);
}

#[test]
fn test_rename_map_not_found_returns_error() {
    let tensors = HashMap::new();
    let rename = HashMap::from([("a.weight".to_string(), "b.weight".to_string())]);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu).with_rename_map(rename);
    // "a.weight" maps to "b.weight" but backend has no tensors → TensorNotFound.
    let err = vb.get(&[1], "a.weight").unwrap_err();
    match err {
        TensorError::TensorNotFound { name } => {
            assert_eq!(name, "b.weight");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

// -- tensor_names tests (#2422) -----------------------------------------------

#[test]
fn test_tensor_names_zeros_backend_returns_empty() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    assert!(vb.tensor_names().is_empty());
}

#[test]
fn test_tensor_names_tensor_map_backend() {
    let vb = VarBuilder::from_tensors(sample_tensors(), DType::F32, &Device::Cpu);
    let mut names = vb.tensor_names();
    names.sort();
    assert_eq!(
        names,
        vec!["decoder.weight", "encoder.bias", "encoder.weight"]
    );
}

#[test]
fn test_tensor_names_empty_map() {
    let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu);
    assert!(vb.tensor_names().is_empty());
}

// -- HuggingFace weight mapping pattern test (#2422) --------------------------

#[test]
fn test_huggingface_granite_docling_pattern() {
    // Simulates loading Granite-Docling weights with HuggingFace naming into
    // an NN model that expects different names. This is the primary use case
    // for #2422 (dpdf VarBuilder weight name mapping).
    let mut hf_tensors = HashMap::new();
    hf_tensors.insert(
        "model.vision_model.encoder.layers.0.self_attn.q_proj.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    hf_tensors.insert(
        "model.vision_model.encoder.layers.0.self_attn.k_proj.weight".to_string(),
        DynTensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    hf_tensors.insert(
        "model.decoder.layers.0.self_attn.q_proj.weight".to_string(),
        DynTensor::new(&[9.0], &[1], &Device::Cpu).unwrap(),
    );

    // Build the rename map from NN names → HF checkpoint names.
    let rename = HashMap::from([
        (
            "vision_encoder.blocks.0.attn.q.weight".to_string(),
            "model.vision_model.encoder.layers.0.self_attn.q_proj.weight".to_string(),
        ),
        (
            "vision_encoder.blocks.0.attn.k.weight".to_string(),
            "model.vision_model.encoder.layers.0.self_attn.k_proj.weight".to_string(),
        ),
        (
            "decoder.layers.0.attn.q.weight".to_string(),
            "model.decoder.layers.0.self_attn.q_proj.weight".to_string(),
        ),
    ]);
    let vb = VarBuilder::from_tensors(hf_tensors, DType::F32, &Device::Cpu).with_rename_map(rename);

    // NN model loads using its own naming convention.
    let q = vb
        .pp("vision_encoder")
        .pp("blocks")
        .pp("0")
        .pp("attn")
        .get(&[2, 2], "q.weight")
        .unwrap();
    assert_eq!(q.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

    let k = vb
        .pp("vision_encoder")
        .pp("blocks")
        .pp("0")
        .pp("attn")
        .get(&[2, 2], "k.weight")
        .unwrap();
    assert_eq!(k.to_flat_vec::<f32>().unwrap(), vec![5.0, 6.0, 7.0, 8.0]);

    let dec_q = vb
        .pp("decoder")
        .pp("layers")
        .pp("0")
        .pp("attn")
        .get(&[1], "q.weight")
        .unwrap();
    assert_eq!(dec_q.to_flat_vec::<f32>().unwrap(), vec![9.0]);
}

#[test]
fn test_tensor_names_returns_raw_backend_keys_with_rename_map() {
    // tensor_names() must return raw backend keys, not mapped keys.
    // This is the weight discovery use case: enumerate checkpoint names.
    let mut tensors = HashMap::new();
    tensors.insert(
        "hf.encoder.weight".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let rename = HashMap::from([(
        "nn.encoder.weight".to_string(),
        "hf.encoder.weight".to_string(),
    )]);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu).with_rename_map(rename);
    // tensor_names() returns the backend key (HF name), not the NN name.
    assert_eq!(vb.tensor_names(), vec!["hf.encoder.weight"]);
}

#[test]
fn test_rename_map_contains_tensor() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "checkpoint.w".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let rename = HashMap::from([("model.w".to_string(), "checkpoint.w".to_string())]);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu).with_rename_map(rename);
    // NN name resolves through rename map → backend has the checkpoint key.
    assert!(vb.contains_tensor("model.w"));
    // Raw checkpoint name doesn't go through mapping → passes through unchanged.
    assert!(vb.contains_tensor("checkpoint.w"));
    // Non-existent key.
    assert!(!vb.contains_tensor("nonexistent"));
}

// == VarBuilder weight loading infrastructure tests ============================
// Tests below cover gaps in construction, tensor loading, backend integration,
// and model-loading patterns for the VarBuilder weight loading system.

// -- A. VarBuilder Construction -----------------------------------------------

#[test]
fn test_varbuilder_from_tensor_map_basic() {
    // Construct from HashMap, verify dtype/device/prefix default state.
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    assert_eq!(vb.dtype(), DType::F32);
    assert_eq!(*vb.device(), Device::Cpu);
    assert_eq!(vb.prefix(), "");
    assert!(!vb.has_name_mapping());
    assert!(vb.precision_policy().is_none());
}

#[test]
fn test_varbuilder_prefix_four_levels() {
    // Deep nesting: model.encoder.layers.0
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let deep = vb.pp("model").pp("encoder").pp("layers").pp("0");
    assert_eq!(deep.prefix(), "model.encoder.layers.0");
}

#[test]
fn test_varbuilder_prefix_numeric_segments() {
    // Numeric string segments for layer indexing (common PyTorch pattern).
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let child = vb.pp("layers").pp("12").pp("attention").pp("3");
    assert_eq!(child.prefix(), "layers.12.attention.3");
}

#[test]
fn test_varbuilder_empty_map_get_returns_error() {
    // Empty backend: any get() call should return TensorNotFound.
    let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu);
    let err = vb.get(&[2], "weight").unwrap_err();
    match err {
        TensorError::TensorNotFound { name } => {
            assert_eq!(name, "weight");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

#[test]
fn test_varbuilder_empty_map_get_unchecked_returns_error() {
    let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu);
    let err = vb.get_unchecked("bias").unwrap_err();
    match err {
        TensorError::TensorNotFound { name } => {
            assert_eq!(name, "bias");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

#[test]
fn test_varbuilder_dtype_override_via_to_dtype() {
    // VarBuilder's dtype propagates to backend calls. Changing dtype via
    // to_dtype should not affect the original VarBuilder.
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let vb_bf16 = vb.to_dtype(DType::BF16);
    assert_eq!(vb.dtype(), DType::F32);
    assert_eq!(vb_bf16.dtype(), DType::BF16);
    // Both can still produce tensors (ZerosBackend always succeeds).
    let t32 = vb.get(&[2], "x").unwrap();
    assert_eq!(t32.dtype(), DType::F32);
    // BF16 zero tensor via ZerosBackend.
    let tbf = vb_bf16.get(&[2], "x").unwrap();
    assert_eq!(tbf.dtype(), DType::BF16);
}

// -- B. Tensor Loading --------------------------------------------------------

#[test]
fn test_get_existing_tensor_preserves_values() {
    let mut map = HashMap::new();
    map.insert(
        "layer.weight".to_string(),
        DynTensor::new(&[0.5, -1.0, 2.5, 3.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let t = vb.pp("layer").get(&[2, 2], "weight").unwrap();
    let data = t.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![0.5, -1.0, 2.5, 3.0]);
}

#[test]
fn test_get_missing_tensor_includes_full_path_in_error() {
    let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu);
    let err = vb
        .pp("encoder")
        .pp("layer")
        .pp("0")
        .get(&[4], "weight")
        .unwrap_err();
    match err {
        TensorError::TensorNotFound { name } => {
            assert_eq!(
                name, "encoder.layer.0.weight",
                "error should include full dot-separated path"
            );
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

#[test]
fn test_get_shape_mismatch_rank_differs() {
    // Shape mismatch where rank differs (requesting 1D but stored is 2D).
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let err = vb.get(&[4], "w").unwrap_err();
    match err {
        TensorError::ShapeMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, vec![4]);
            assert_eq!(actual, vec![2, 2]);
        }
        other => panic!("expected ShapeMismatch, got: {other:?}"),
    }
}

#[test]
fn test_get_shape_mismatch_same_rank_different_dims() {
    // Same rank [2, 3] vs [3, 2] — transposed shape.
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
fn test_get_scalar_tensor() {
    // 0-D scalar tensor via VarBuilder.
    let mut map = HashMap::new();
    map.insert(
        "scale".to_string(),
        DynTensor::new(&[42.0], &[], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let t = vb.get(&[], "scale").unwrap();
    assert_eq!(t.dims(), &[] as &[usize]);
    let data = t.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![42.0]);
}

#[test]
fn test_get_large_tensor_shape() {
    // Verify large shape is accepted (no allocation, just shape check).
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let t = vb.get(&[32, 128, 768], "embedding").unwrap();
    assert_eq!(t.dims(), &[32, 128, 768]);
}

// -- C. Backend Integration ---------------------------------------------------

#[test]
fn test_custom_backend_via_from_backend() {
    use std::sync::Arc;

    struct ConstBackend {
        val: f64,
    }
    impl super::TensorBackend for ConstBackend {
        fn get(
            &self,
            dims: &[usize],
            _name: &str,
            dtype: DType,
            device: &Device,
        ) -> crate::Result<DynTensor> {
            DynTensor::full(dims, self.val, dtype, device)
        }
        fn get_unchecked(
            &self,
            _name: &str,
            dtype: DType,
            device: &Device,
        ) -> crate::Result<DynTensor> {
            DynTensor::full(&[], self.val, dtype, device)
        }
        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
        fn tensor_names(&self) -> Vec<String> {
            vec!["const_weight".to_string()]
        }
    }

    let backend = Arc::new(ConstBackend { val: 7.0 });
    let vb = VarBuilder::from_backend(backend, DType::F32, Device::Cpu);
    let t = vb.get(&[3], "anything").unwrap();
    let data = t.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![7.0, 7.0, 7.0]);
    assert!(vb.contains_tensor("whatever"));
    assert_eq!(vb.tensor_names(), vec!["const_weight"]);
}

#[test]
fn test_zeros_backend_returns_correct_dtype() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let t = vb.get(&[2, 3], "w").unwrap();
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.dims(), &[2, 3]);
    let data = t.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_backend_prefix_scoping_isolates_keys() {
    // Two different prefixes access different tensors from the same backend.
    let mut map = HashMap::new();
    map.insert(
        "encoder.weight".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    map.insert(
        "decoder.weight".to_string(),
        DynTensor::new(&[3.0, 4.0], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);

    let enc = vb.pp("encoder");
    let dec = vb.pp("decoder");

    let enc_w = enc.get(&[2], "weight").unwrap();
    let dec_w = dec.get(&[2], "weight").unwrap();

    assert_eq!(enc_w.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(dec_w.to_flat_vec::<f32>().unwrap(), vec![3.0, 4.0]);

    // Cross-prefix access should fail.
    assert!(!enc.contains_tensor("decoder.weight"));
    assert!(!dec.contains_tensor("encoder.weight"));
}

#[test]
fn test_tensor_map_backend_contains_false_for_missing() {
    let mut map = HashMap::new();
    map.insert(
        "a".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    assert!(vb.contains_tensor("a"));
    assert!(!vb.contains_tensor("b"));
    assert!(!vb.contains_tensor(""));
}

#[test]
fn test_zeros_backend_tensor_names_empty() {
    // ZerosBackend has infinite key space, so tensor_names() returns empty.
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    assert!(vb.tensor_names().is_empty());
}

#[test]
fn test_tensor_map_backend_tensor_names_complete() {
    let mut map = HashMap::new();
    for i in 0..5 {
        map.insert(
            format!("layer{i}.weight"),
            DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
        );
    }
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let mut names = vb.tensor_names();
    names.sort();
    assert_eq!(
        names,
        vec![
            "layer0.weight",
            "layer1.weight",
            "layer2.weight",
            "layer3.weight",
            "layer4.weight"
        ]
    );
}

// -- D. Model Loading Patterns ------------------------------------------------

#[test]
fn test_load_linear_weight_and_bias_values() {
    // Verify Linear layer loads weight [out, in] and bias [out] correctly.
    let mut map = HashMap::new();
    map.insert(
        "proj.weight".to_string(),
        DynTensor::new(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[3, 3],
            &Device::Cpu,
        )
        .unwrap(),
    );
    map.insert(
        "proj.bias".to_string(),
        DynTensor::new(&[100.0, 200.0, 300.0], &[3], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let t_weight = vb.pp("proj").get(&[3, 3], "weight").unwrap();
    let t_bias = vb.pp("proj").get(&[3], "bias").unwrap();
    assert_eq!(t_weight.dims(), &[3, 3]);
    assert_eq!(t_bias.dims(), &[3]);
    assert_eq!(
        t_bias.to_flat_vec::<f32>().unwrap(),
        vec![100.0, 200.0, 300.0]
    );
}

#[test]
fn test_load_conv1d_weight_shape() {
    // Conv1d weight: [out_channels, in_channels/groups, kernel_size]
    let mut map = HashMap::new();
    map.insert(
        "conv.weight".to_string(),
        DynTensor::zeros(&[16, 8, 3], DType::F32, &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let w = vb.pp("conv").get(&[16, 8, 3], "weight").unwrap();
    assert_eq!(w.dims(), &[16, 8, 3]);
}

#[test]
fn test_load_layernorm_weight_and_bias() {
    // LayerNorm: weight [dim], bias [dim]
    let dim = 512;
    let mut map = HashMap::new();
    map.insert(
        "norm.weight".to_string(),
        DynTensor::ones(&[dim], DType::F32, &Device::Cpu).unwrap(),
    );
    map.insert(
        "norm.bias".to_string(),
        DynTensor::zeros(&[dim], DType::F32, &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let w = vb.pp("norm").get(&[dim], "weight").unwrap();
    let b = vb.pp("norm").get(&[dim], "bias").unwrap();
    assert_eq!(w.dims(), &[dim]);
    assert_eq!(b.dims(), &[dim]);
    // Verify ones
    let wv = w.to_flat_vec::<f32>().unwrap();
    assert!(wv.iter().all(|&v| (v - 1.0).abs() < 1e-6));
}

#[test]
fn test_load_embedding_weight_shape() {
    // Embedding: weight [vocab_size, embedding_dim]
    let vocab = 32000;
    let dim = 768;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let w = vb.pp("embed_tokens").get(&[vocab, dim], "weight").unwrap();
    assert_eq!(w.dims(), &[vocab, dim]);
}

// -- E. Additional edge cases and patterns ------------------------------------

#[test]
fn test_multiple_pp_branches_share_backend() {
    // Two independent pp() branches from the same root share the backend.
    let mut map = HashMap::new();
    map.insert(
        "a.w".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    map.insert(
        "b.w".to_string(),
        DynTensor::new(&[2.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let a = vb.pp("a");
    let b = vb.pp("b");

    let ta = a.get(&[1], "w").unwrap();
    let tb = b.get(&[1], "w").unwrap();
    assert_eq!(ta.to_flat_vec::<f32>().unwrap(), vec![1.0]);
    assert_eq!(tb.to_flat_vec::<f32>().unwrap(), vec![2.0]);
}

#[test]
fn test_pp_does_not_mutate_parent() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let _child = vb.pp("child");
    // Parent prefix should remain empty.
    assert_eq!(vb.prefix(), "");
}

#[test]
fn test_dtype_propagates_through_pp() {
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let child = vb.pp("encoder").pp("layer0");
    assert_eq!(child.dtype(), DType::BF16);
}

#[test]
fn test_device_propagates_through_pp() {
    let vb = VarBuilder::zeros(DType::F32, &Device::metal());
    let child = vb.pp("decoder");
    assert_eq!(*child.device(), Device::metal());
}

#[test]
fn test_name_mapping_not_found_returns_mapped_key_in_error() {
    // When name mapping transforms the key, the error should report the
    // mapped (backend) key, not the NN model key.
    let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu)
        .with_name_mapping(|name| name.replace("model", "checkpoint"));
    let err = vb.pp("model").get(&[1], "weight").unwrap_err();
    match err {
        TensorError::TensorNotFound { name } => {
            assert_eq!(
                name, "checkpoint.weight",
                "error should contain the mapped key"
            );
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

#[test]
fn test_with_weight_name_mapper_hf_to_nn() {
    use crate::var_builder::HfToNnMapper;

    let mut tensors = HashMap::new();
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    let mapper = HfToNnMapper::new()
        .with_prefix_rule("model.layers", "encoder.layer")
        .with_segment_rule("self_attn", "attention")
        .with_segment_rule("q_proj", "q");
    let vb =
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu).with_weight_name_mapper(mapper);
    let t = vb
        .pp("encoder")
        .pp("layer")
        .pp("0")
        .pp("attention")
        .get(&[2, 2], "q.weight")
        .unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_nan_in_multiple_positions() {
    // Multiple NaN values should be counted in NonFiniteData error.
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(
            &[f32::NAN, 1.0, f32::NAN, 2.0, f32::NAN],
            &[5],
            &Device::Cpu,
        )
        .unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let err = vb.get(&[5], "w").unwrap_err();
    match err {
        TensorError::NonFiniteData { count, .. } => {
            assert_eq!(count, 3, "should count all 3 NaN values");
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

#[test]
fn test_mixed_nan_and_inf() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(
            &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0],
            &[4],
            &Device::Cpu,
        )
        .unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let err = vb.get(&[4], "w").unwrap_err();
    match err {
        TensorError::NonFiniteData { count, .. } => {
            assert_eq!(count, 3, "NaN + Inf + -Inf = 3 non-finite values");
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

#[test]
fn test_as_ref_varbuilder_owned_and_borrowed() {
    // VarBuilder implements AsRef<VarBuilder> — verify both owned and & work.
    fn accepts_asref(vb: impl AsRef<VarBuilder>) -> String {
        vb.as_ref().prefix()
    }
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu).pp("test");
    assert_eq!(accepts_asref(&vb), "test");
    assert_eq!(accepts_asref(vb), "test");
}

#[test]
fn test_varbuilder_is_send_sync() {
    // VarBuilder uses Arc<dyn TensorBackend> which requires Send+Sync.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<VarBuilder>();
}

#[test]
fn test_tensor_map_with_single_element_tensor() {
    let mut map = HashMap::new();
    map.insert(
        "scalar".to_string(),
        DynTensor::new(&[99.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let t = vb.get(&[1], "scalar").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![99.0]);
}

#[test]
fn test_prefix_mapping_no_match_passthrough() {
    // When no prefix matches, the key should pass through unchanged.
    let mut map = HashMap::new();
    map.insert(
        "original.weight".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
        .with_prefix_mapping(&[("unrelated", "other")]);
    // "original.weight" does not match "unrelated" prefix, so it passes through.
    let t = vb.get(&[1], "original.weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0]);
}
