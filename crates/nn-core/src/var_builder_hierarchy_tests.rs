// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! VarBuilder hierarchy tests.
//!
//! Exercises hierarchical prefix navigation (`pp()`), tensor lookup through
//! nested paths, missing tensor error messages, name mapping propagation,
//! and `verify_mapper_coverage` integration.
//!
//! Part of #4186.

use std::collections::HashMap;

use crate::dyn_tensor::DynTensor;
use crate::var_builder::{verify_mapper_coverage, HfToNnMapper, VarBuilder};
use crate::{DType, Device, TensorError};

// ---------------------------------------------------------------------------
// A. Push prefix creates nested paths
// ---------------------------------------------------------------------------

#[test]
fn test_pp_single_level_prefix() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let child = vb.pp("encoder");
    assert_eq!(child.prefix(), "encoder");
}

#[test]
fn test_pp_two_levels_dot_separated() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let child = vb.pp("encoder").pp("layer0");
    assert_eq!(child.prefix(), "encoder.layer0");
}

#[test]
fn test_pp_deep_nesting_five_levels() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let deep = vb
        .pp("model")
        .pp("encoder")
        .pp("layers")
        .pp("0")
        .pp("self_attn");
    assert_eq!(deep.prefix(), "model.encoder.layers.0.self_attn");
}

#[test]
fn test_pp_empty_segments_skipped() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let child = vb.pp("").pp("encoder").pp("").pp("layer").pp("");
    assert_eq!(child.prefix(), "encoder.layer");
}

#[test]
fn test_pp_does_not_modify_parent() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let _child = vb.pp("child");
    let _grandchild = vb.pp("child").pp("grandchild");
    assert_eq!(vb.prefix(), "");
}

// ---------------------------------------------------------------------------
// B. Get tensor by hierarchical name
// ---------------------------------------------------------------------------

#[test]
fn test_hierarchical_get_resolves_dotted_key() {
    let mut map = HashMap::new();
    map.insert(
        "encoder.layer.0.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);

    let t = vb
        .pp("encoder")
        .pp("layer")
        .pp("0")
        .get(&[2, 2], "weight")
        .unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(t.dims(), &[2, 2]);
}

#[test]
fn test_hierarchical_get_unchecked_skips_shape() {
    let mut map = HashMap::new();
    map.insert(
        "decoder.blocks.3.bias".to_string(),
        DynTensor::new(&[0.5, -0.5], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);

    let t = vb
        .pp("decoder")
        .pp("blocks")
        .pp("3")
        .get_unchecked("bias")
        .unwrap();
    assert_eq!(t.dims(), &[2]);
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![0.5, -0.5]);
}

#[test]
fn test_contains_tensor_through_hierarchy() {
    let mut map = HashMap::new();
    map.insert(
        "a.b.c.weight".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);

    assert!(vb.pp("a").pp("b").pp("c").contains_tensor("weight"));
    assert!(!vb.pp("a").pp("b").pp("c").contains_tensor("bias"));
    assert!(!vb.pp("a").pp("b").contains_tensor("weight"));
}

// ---------------------------------------------------------------------------
// C. Missing tensor returns appropriate error
// ---------------------------------------------------------------------------

#[test]
fn test_missing_tensor_error_includes_full_hierarchical_path() {
    let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu);
    let err = vb
        .pp("model")
        .pp("encoder")
        .pp("layer.0")
        .pp("attn")
        .get(&[4, 4], "q_proj.weight")
        .unwrap_err();

    match err {
        TensorError::TensorNotFound { name } => {
            assert_eq!(name, "model.encoder.layer.0.attn.q_proj.weight");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

#[test]
fn test_missing_tensor_get_unchecked_error_path() {
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
fn test_shape_mismatch_through_hierarchy() {
    let mut map = HashMap::new();
    map.insert(
        "layer.weight".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);

    let err = vb.pp("layer").get(&[3], "weight").unwrap_err();
    match err {
        TensorError::ShapeMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, vec![3]);
            assert_eq!(actual, vec![2]);
        }
        other => panic!("expected ShapeMismatch, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// D. Name mapping functions through hierarchy
// ---------------------------------------------------------------------------

#[test]
fn test_name_mapping_transforms_hierarchical_key() {
    let mut map = HashMap::new();
    map.insert(
        "hf.layers.0.attn.weight".to_string(),
        DynTensor::new(&[9.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
        .with_name_mapping(|name| name.replace("nn.layers", "hf.layers"));

    let t = vb
        .pp("nn")
        .pp("layers")
        .pp("0")
        .pp("attn")
        .get(&[1], "weight")
        .unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![9.0]);
}

#[test]
fn test_name_mapping_propagates_through_pp_chain() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu).with_name_mapping(str::to_uppercase);

    assert!(vb.has_name_mapping());
    let child = vb.pp("encoder").pp("layer").pp("0");
    assert!(child.has_name_mapping());
}

#[test]
fn test_prefix_mapping_through_hierarchy() {
    let mut map = HashMap::new();
    map.insert(
        "checkpoint.encoder.weight".to_string(),
        DynTensor::new(&[7.0, 8.0], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
        .with_prefix_mapping(&[("model.encoder", "checkpoint.encoder")]);

    let t = vb.pp("model").pp("encoder").get(&[2], "weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![7.0, 8.0]);
}

#[test]
fn test_rename_map_through_hierarchy() {
    let mut map = HashMap::new();
    map.insert(
        "actual.key".to_string(),
        DynTensor::new(&[3.14], &[1], &Device::Cpu).unwrap(),
    );
    let rename = HashMap::from([("logical.path.weight".to_string(), "actual.key".to_string())]);
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu).with_rename_map(rename);

    let t = vb.pp("logical").pp("path").get(&[1], "weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![3.14]);
}

// ---------------------------------------------------------------------------
// E. Weight coverage verification (verify_mapper_coverage)
// ---------------------------------------------------------------------------

#[test]
fn test_verify_coverage_all_mapped() {
    let mapper = HfToNnMapper::new().with_prefix_rule("model", "m");
    let checkpoint = vec!["model.w".to_string(), "model.b".to_string()];
    let nn = vec!["m.w".to_string(), "m.b".to_string()];

    let missing = verify_mapper_coverage(&nn, &checkpoint, &mapper);
    assert!(
        missing.is_empty(),
        "all names should be covered, got: {missing:?}"
    );
}

#[test]
fn test_verify_coverage_missing_names_reported() {
    let mapper = HfToNnMapper::new();
    let checkpoint = vec!["a".to_string()];
    let nn = vec!["a".to_string(), "b".to_string(), "c".to_string()];

    let missing = verify_mapper_coverage(&nn, &checkpoint, &mapper);
    assert_eq!(missing.len(), 2);
    assert!(missing.contains(&"b".to_string()));
    assert!(missing.contains(&"c".to_string()));
}

#[test]
fn test_verify_coverage_empty_nn_names() {
    let mapper = HfToNnMapper::new();
    let checkpoint = vec!["some.weight".to_string()];
    let nn: Vec<String> = vec![];

    let missing = verify_mapper_coverage(&nn, &checkpoint, &mapper);
    assert!(missing.is_empty());
}

#[test]
fn test_verify_coverage_complex_decoder_mapper() {
    let mapper = HfToNnMapper::decoder_transformer();
    let checkpoint = vec![
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        "model.layers.0.mlp.gate_proj.weight".to_string(),
    ];
    let nn = vec![
        "layers.0.attn.q.weight".to_string(),
        "layers.0.attn.k.weight".to_string(),
        "layers.0.mlp.gate.weight".to_string(),
    ];

    let missing = verify_mapper_coverage(&nn, &checkpoint, &mapper);
    assert!(
        missing.is_empty(),
        "decoder_transformer mapper should cover all standard names, missing: {missing:?}"
    );
}

// ---------------------------------------------------------------------------
// F. HfToNnMapper through VarBuilder hierarchy
// ---------------------------------------------------------------------------

#[test]
fn test_hf_mapper_with_varbuilder_hierarchy() {
    let mut map = HashMap::new();
    map.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    let mapper = HfToNnMapper::decoder_transformer();
    let vb =
        VarBuilder::from_tensors(map, DType::F32, &Device::Cpu).with_weight_name_mapper(mapper);

    let t = vb
        .pp("layers")
        .pp("0")
        .pp("attn")
        .get(&[2, 2], "q.weight")
        .unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

// ---------------------------------------------------------------------------
// G. Dtype and device propagation through hierarchy
// ---------------------------------------------------------------------------

#[test]
fn test_dtype_propagates_through_full_hierarchy() {
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let deep = vb.pp("a").pp("b").pp("c").pp("d");
    assert_eq!(deep.dtype(), DType::BF16);
}

#[test]
fn test_to_dtype_preserves_prefix_hierarchy() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu)
        .pp("encoder")
        .pp("layer");
    let vb2 = vb.to_dtype(DType::F16);
    assert_eq!(vb2.prefix(), "encoder.layer");
    assert_eq!(vb2.dtype(), DType::F16);
    // Original unchanged.
    assert_eq!(vb.dtype(), DType::F32);
}

// ---------------------------------------------------------------------------
// H. Safetensors loaded into VarBuilder hierarchy
// ---------------------------------------------------------------------------

#[test]
fn test_safetensors_loaded_into_varbuilder_hierarchy() {
    use crate::dyn_tensor::{load_safetensors_from_bytes, tensors_to_safetensors_bytes};

    let mut tensors = HashMap::new();
    tensors.insert(
        "model.encoder.layers.0.weight".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "model.encoder.layers.0.bias".to_string(),
        DynTensor::new(&[0.1, 0.2], &[2], &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "model.decoder.weight".to_string(),
        DynTensor::new(&[3.0, 4.0, 5.0], &[3], &Device::Cpu).unwrap(),
    );

    let bytes = tensors_to_safetensors_bytes(&tensors).unwrap();
    let loaded = load_safetensors_from_bytes(&bytes).unwrap();
    let vb = VarBuilder::from_tensors(loaded, DType::F32, &Device::Cpu);

    let w = vb
        .pp("model")
        .pp("encoder")
        .pp("layers")
        .pp("0")
        .get(&[2], "weight")
        .unwrap();
    let b = vb
        .pp("model")
        .pp("encoder")
        .pp("layers")
        .pp("0")
        .get(&[2], "bias")
        .unwrap();
    let dec = vb.pp("model").pp("decoder").get(&[3], "weight").unwrap();

    assert_eq!(w.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(b.to_flat_vec::<f32>().unwrap(), vec![0.1, 0.2]);
    assert_eq!(dec.to_flat_vec::<f32>().unwrap(), vec![3.0, 4.0, 5.0]);
}
