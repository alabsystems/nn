// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for VarBuilder weight loading and name mapping.
//!
//! Covers: hierarchical prefix stacking (triple+), weight name mapper coverage
//! verification, HfToNnMapper pattern matching edge cases, missing weight error
//! messages with mapped keys, shape mismatch on get(), zero-initialized backend
//! behavior, tensor map backend with pre-loaded tensors, contains_tensor() for
//! present/absent keys, and VarBuilder clone/sharing semantics.
//!
//! Part of #4495.

use std::collections::HashMap;
use std::sync::Arc;

use crate::dyn_tensor::DynTensor;
use crate::var_builder::{
    verify_mapper_coverage, HfToNnMapper, TensorBackend, TensorMapBackend, VarBuilder,
    WeightNameMapper, ZerosBackend,
};
use crate::{DType, Device, TensorError};

// ===========================================================================
// A. Hierarchical prefix stacking (triple and beyond)
// ===========================================================================

#[test]
fn test_pp_triple_stacking_abc() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let deep = vb.pp("a").pp("b").pp("c");
    assert_eq!(deep.prefix(), "a.b.c");
}

#[test]
fn test_pp_triple_stacking_resolves_tensor_key() {
    let mut map = HashMap::new();
    map.insert(
        "a.b.c.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let t = vb.pp("a").pp("b").pp("c").get(&[3], "weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_pp_six_level_stacking() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let deep = vb
        .pp("model")
        .pp("transformer")
        .pp("encoder")
        .pp("layers")
        .pp("0")
        .pp("self_attn");
    assert_eq!(
        deep.prefix(),
        "model.transformer.encoder.layers.0.self_attn"
    );
}

#[test]
fn test_pp_stacking_with_numeric_indices() {
    let mut map = HashMap::new();
    map.insert(
        "layers.0.blocks.1.heads.2.weight".to_string(),
        DynTensor::new(&[42.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let t = vb
        .pp("layers")
        .pp("0")
        .pp("blocks")
        .pp("1")
        .pp("heads")
        .pp("2")
        .get(&[1], "weight")
        .unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![42.0]);
}

#[test]
fn test_pp_stacking_with_empty_segments_skipped() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let deep = vb.pp("a").pp("").pp("b").pp("").pp("c");
    assert_eq!(deep.prefix(), "a.b.c");
}

#[test]
fn test_pp_stacking_sibling_branches_independent() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let branch_a = vb.pp("a").pp("b").pp("c");
    let branch_b = vb.pp("x").pp("y").pp("z");

    assert_eq!(branch_a.prefix(), "a.b.c");
    assert_eq!(branch_b.prefix(), "x.y.z");
    // Parent is untouched.
    assert_eq!(vb.prefix(), "");
}

// ===========================================================================
// B. Weight name mapper coverage verification
// ===========================================================================

#[test]
fn test_verify_mapper_coverage_with_segment_rules() {
    let mapper = HfToNnMapper::new()
        .with_segment_rule("self_attn", "attn")
        .with_segment_rule("q_proj", "q");

    let checkpoint_names = vec![
        "layer.0.self_attn.q_proj.weight".to_string(),
        "layer.0.self_attn.q_proj.bias".to_string(),
    ];
    let nn_names = vec![
        "layer.0.attn.q.weight".to_string(),
        "layer.0.attn.q.bias".to_string(),
    ];

    let missing = verify_mapper_coverage(&nn_names, &checkpoint_names, &mapper);
    assert!(
        missing.is_empty(),
        "segment rules should map all names, got missing: {missing:?}"
    );
}

#[test]
fn test_verify_mapper_coverage_with_suffix_rules() {
    let mapper = HfToNnMapper::new().with_suffix_rule("_proj", &["q", "k", "v"]);

    let checkpoint_names = vec![
        "layer.q_proj.weight".to_string(),
        "layer.k_proj.weight".to_string(),
        "layer.v_proj.weight".to_string(),
    ];
    let nn_names = vec![
        "layer.q.weight".to_string(),
        "layer.k.weight".to_string(),
        "layer.v.weight".to_string(),
    ];

    let missing = verify_mapper_coverage(&nn_names, &checkpoint_names, &mapper);
    assert!(
        missing.is_empty(),
        "suffix rules should cover all, got: {missing:?}"
    );
}

#[test]
fn test_verify_mapper_coverage_missing_due_to_no_rule() {
    let mapper = HfToNnMapper::new().with_segment_rule("self_attn", "attn");

    // "mlp_gate" has no rule, so "layer.mlp_gate.weight" maps to itself.
    let checkpoint_names = vec![
        "layer.self_attn.weight".to_string(),
        "layer.feed_forward.weight".to_string(),
    ];
    let nn_names = vec![
        "layer.attn.weight".to_string(),
        "layer.mlp_gate.weight".to_string(), // no rule for this
    ];

    let missing = verify_mapper_coverage(&nn_names, &checkpoint_names, &mapper);
    assert_eq!(missing.len(), 1);
    assert_eq!(missing[0], "layer.mlp_gate.weight");
}

#[test]
fn test_verify_mapper_coverage_duplicate_nn_names() {
    let mapper = HfToNnMapper::new();
    let checkpoint_names = vec!["w".to_string()];
    let nn_names = vec!["w".to_string(), "w".to_string()];

    // Both instances should be covered (identity mapper).
    let missing = verify_mapper_coverage(&nn_names, &checkpoint_names, &mapper);
    assert!(missing.is_empty());
}

// ===========================================================================
// C. HfToNnMapper pattern matching edge cases
// ===========================================================================

#[test]
fn test_hf_mapper_prefix_and_segment_combined_order() {
    // Prefix is applied first, then segment rules.
    let mapper = HfToNnMapper::new()
        .with_prefix_rule("hf.model", "nn")
        .with_segment_rule("self_attn", "attn");

    // "nn.0.attn.weight" -> prefix: "hf.model.0.attn.weight" -> segment: "hf.model.0.self_attn.weight"
    assert_eq!(
        mapper.map_name("nn.0.attn.weight"),
        "hf.model.0.self_attn.weight"
    );
}

#[test]
fn test_hf_mapper_prefix_rule_exact_match_entire_name() {
    let mapper = HfToNnMapper::new().with_prefix_rule("replacement", "original");
    // "original" with no trailing dot — the entire name is the prefix.
    assert_eq!(mapper.map_name("original"), "replacement");
}

#[test]
fn test_hf_mapper_segment_rule_weight_and_bias_unchanged() {
    // "weight" and "bias" are standard terminal segments and should not be
    // affected by rules targeting other segments.
    let mapper = HfToNnMapper::new()
        .with_segment_rule("self_attn", "attn")
        .with_segment_rule("q_proj", "q");
    assert_eq!(mapper.map_name("layer.weight"), "layer.weight");
    assert_eq!(mapper.map_name("layer.bias"), "layer.bias");
}

#[test]
fn test_hf_mapper_segment_rule_same_segment_twice_in_path() {
    // If "attn" appears twice, both should be mapped.
    let mapper = HfToNnMapper::new().with_segment_rule("self_attn", "attn");
    assert_eq!(
        mapper.map_name("attn.layers.attn.weight"),
        "self_attn.layers.self_attn.weight"
    );
}

#[test]
fn test_hf_mapper_suffix_rule_does_not_affect_terminal_weight() {
    // The suffix rule for "q" -> "q_proj" should NOT affect "weight".
    let mapper = HfToNnMapper::new().with_suffix_rule("_proj", &["q"]);
    assert_eq!(mapper.map_name("layer.weight"), "layer.weight");
}

#[test]
fn test_hf_mapper_chained_prefix_and_suffix_rules() {
    let mapper = HfToNnMapper::new()
        .with_prefix_rule("model.layers", "layers")
        .with_suffix_rule("_proj", &["q", "k", "v"]);

    assert_eq!(
        mapper.map_name("layers.0.q.weight"),
        "model.layers.0.q_proj.weight"
    );
}

#[test]
fn test_hf_mapper_exact_override_takes_priority_over_prefix_and_segment() {
    let mut overrides = HashMap::new();
    overrides.insert(
        "custom.special".to_string(),
        "totally.different.key".to_string(),
    );

    let mapper = HfToNnMapper::new()
        .with_prefix_rule("model", "custom")
        .with_segment_rule("replaced", "special")
        .with_exact_overrides(overrides);

    // Exact override should win.
    assert_eq!(mapper.map_name("custom.special"), "totally.different.key");
    // Non-override still goes through rules.
    assert_eq!(mapper.map_name("custom.other"), "model.other");
}

#[test]
fn test_hf_mapper_clone_preserves_rules() {
    let mapper = HfToNnMapper::new()
        .with_prefix_rule("model", "m")
        .with_segment_rule("self_attn", "attn");

    let cloned = mapper;
    assert_eq!(
        cloned.map_name("m.0.attn.weight"),
        "model.0.self_attn.weight"
    );
}

// ===========================================================================
// D. Missing weight error messages
// ===========================================================================

#[test]
fn test_missing_weight_error_includes_mapped_key() {
    let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu)
        .with_name_mapping(|name| name.replace("nn_prefix", "hf_prefix"));

    let err = vb
        .pp("nn_prefix")
        .pp("encoder")
        .get(&[4], "weight")
        .unwrap_err();

    match err {
        TensorError::TensorNotFound { name } => {
            // The error should contain the MAPPED key (what the backend saw).
            assert_eq!(name, "hf_prefix.encoder.weight");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

#[test]
fn test_missing_weight_error_with_hf_mapper() {
    let mapper = HfToNnMapper::new()
        .with_prefix_rule("model.layers", "layers")
        .with_segment_rule("self_attn", "attn");

    let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu)
        .with_weight_name_mapper(mapper);

    let err = vb
        .pp("layers")
        .pp("0")
        .pp("attn")
        .get(&[4, 4], "q.weight")
        .unwrap_err();

    match err {
        TensorError::TensorNotFound { name } => {
            // Mapped to HF naming convention.
            assert_eq!(name, "model.layers.0.self_attn.q.weight");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

#[test]
fn test_missing_weight_error_get_unchecked_with_mapping() {
    let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu)
        .with_name_mapping(|name| format!("checkpoint.{name}"));

    let err = vb.pp("layer").get_unchecked("bias").unwrap_err();
    match err {
        TensorError::TensorNotFound { name } => {
            assert_eq!(name, "checkpoint.layer.bias");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

// ===========================================================================
// E. Shape mismatch on get()
// ===========================================================================

#[test]
fn test_shape_mismatch_rank_differs_with_prefix() {
    let mut map = HashMap::new();
    map.insert(
        "block.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);

    // Requesting rank-1 but stored as rank-2.
    let err = vb.pp("block").get(&[4], "weight").unwrap_err();
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
fn test_shape_mismatch_higher_rank() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);

    // Requesting rank-3 but stored as rank-2.
    let err = vb.get(&[1, 2, 3], "w").unwrap_err();
    match err {
        TensorError::ShapeMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, vec![1, 2, 3]);
            assert_eq!(actual, vec![2, 3]);
        }
        other => panic!("expected ShapeMismatch, got: {other:?}"),
    }
}

#[test]
fn test_shape_mismatch_with_name_mapping() {
    // Even with name mapping, shape mismatch should be reported.
    let mut map = HashMap::new();
    map.insert(
        "mapped.w".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
        .with_name_mapping(|n| n.replace("original", "mapped"));

    let err = vb.get(&[3], "original.w").unwrap_err();
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

#[test]
fn test_shape_mismatch_scalar_vs_vector() {
    let mut map = HashMap::new();
    map.insert(
        "s".to_string(),
        DynTensor::new(&[1.0], &[], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);

    // Requesting [1] but stored as scalar [].
    let err = vb.get(&[1], "s").unwrap_err();
    match err {
        TensorError::ShapeMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, vec![1]);
            assert_eq!(actual, Vec::<usize>::new());
        }
        other => panic!("expected ShapeMismatch, got: {other:?}"),
    }
}

// ===========================================================================
// F. Zero-initialized backend
// ===========================================================================

#[test]
fn test_zeros_backend_returns_zeros_for_any_shape() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    for shape in &[&[1][..], &[2, 3], &[4, 5, 6], &[]] {
        let t = vb.get(shape, "any").unwrap();
        assert_eq!(t.dims(), *shape);
        let data = t.to_flat_vec::<f32>().unwrap();
        assert!(
            data.iter().all(|&v| v == 0.0),
            "all values should be zero for shape {shape:?}"
        );
    }
}

#[test]
fn test_zeros_backend_with_prefix_still_returns_zeros() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let deep = vb.pp("model").pp("encoder").pp("layers").pp("0");
    let t = deep.get(&[4, 4], "weight").unwrap();
    assert_eq!(t.dims(), &[4, 4]);
    let data = t.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_zeros_backend_get_unchecked_always_scalar() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let t = vb.pp("anything").get_unchecked("weight").unwrap();
    assert_eq!(t.dims(), &[] as &[usize]);
}

#[test]
fn test_zeros_backend_different_dtypes() {
    for dtype in &[DType::F32, DType::BF16, DType::F16] {
        let vb = VarBuilder::zeros(*dtype, &Device::Cpu);
        let t = vb.get(&[2, 3], "w").unwrap();
        assert_eq!(t.dtype(), *dtype);
        assert_eq!(t.dims(), &[2, 3]);
    }
}

#[test]
fn test_zeros_backend_tensor_names_always_empty() {
    let backend = ZerosBackend;
    assert!(backend.tensor_names().is_empty());
}

// ===========================================================================
// G. Tensor map backend with pre-loaded tensors
// ===========================================================================

#[test]
fn test_tensor_map_backend_multiple_tensors_different_shapes() {
    let mut map = HashMap::new();
    map.insert(
        "scalar".to_string(),
        DynTensor::new(&[1.0], &[], &Device::Cpu).unwrap(),
    );
    map.insert(
        "vector".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap(),
    );
    map.insert(
        "matrix".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );
    map.insert(
        "rank3".to_string(),
        DynTensor::zeros(&[2, 3, 4], DType::F32, &Device::Cpu).unwrap(),
    );

    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);

    let s = vb.get(&[], "scalar").unwrap();
    assert_eq!(s.dims(), &[] as &[usize]);

    let v = vb.get(&[3], "vector").unwrap();
    assert_eq!(v.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);

    let m = vb.get(&[2, 2], "matrix").unwrap();
    assert_eq!(m.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

    let r3 = vb.get(&[2, 3, 4], "rank3").unwrap();
    assert_eq!(r3.dims(), &[2, 3, 4]);
}

#[test]
fn test_tensor_map_backend_preserves_negative_values() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[-1.5, 0.0, 1.5, -100.0], &[4], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let data = vb.get(&[4], "w").unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![-1.5, 0.0, 1.5, -100.0]);
}

#[test]
fn test_tensor_map_backend_hierarchical_keys() {
    let mut map = HashMap::new();
    map.insert(
        "encoder.layers.0.attn.q.weight".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    map.insert(
        "encoder.layers.0.attn.k.weight".to_string(),
        DynTensor::new(&[3.0, 4.0], &[2], &Device::Cpu).unwrap(),
    );
    map.insert(
        "encoder.layers.1.attn.q.weight".to_string(),
        DynTensor::new(&[5.0, 6.0], &[2], &Device::Cpu).unwrap(),
    );

    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);

    let q0 = vb
        .pp("encoder")
        .pp("layers")
        .pp("0")
        .pp("attn")
        .pp("q")
        .get(&[2], "weight")
        .unwrap();
    let k0 = vb
        .pp("encoder")
        .pp("layers")
        .pp("0")
        .pp("attn")
        .pp("k")
        .get(&[2], "weight")
        .unwrap();
    let q1 = vb
        .pp("encoder")
        .pp("layers")
        .pp("1")
        .pp("attn")
        .pp("q")
        .get(&[2], "weight")
        .unwrap();

    assert_eq!(q0.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(k0.to_flat_vec::<f32>().unwrap(), vec![3.0, 4.0]);
    assert_eq!(q1.to_flat_vec::<f32>().unwrap(), vec![5.0, 6.0]);
}

#[test]
fn test_tensor_map_backend_direct_get_validates_shape() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu).unwrap(),
    );
    let backend = TensorMapBackend::new(map);

    // Correct shape succeeds.
    let t = backend.get(&[2, 3], "w", DType::F32, &Device::Cpu).unwrap();
    assert_eq!(t.dims(), &[2, 3]);

    // Wrong shape fails.
    let err = backend
        .get(&[3, 2], "w", DType::F32, &Device::Cpu)
        .unwrap_err();
    matches!(err, TensorError::ShapeMismatch { .. });
}

// ===========================================================================
// H. contains_tensor() for present and absent keys
// ===========================================================================

#[test]
fn test_contains_tensor_present_key() {
    let mut map = HashMap::new();
    map.insert(
        "encoder.weight".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);

    assert!(vb.pp("encoder").contains_tensor("weight"));
}

#[test]
fn test_contains_tensor_absent_key() {
    let mut map = HashMap::new();
    map.insert(
        "encoder.weight".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);

    assert!(!vb.pp("encoder").contains_tensor("bias"));
    assert!(!vb.pp("decoder").contains_tensor("weight"));
}

#[test]
fn test_contains_tensor_with_name_mapping() {
    let mut map = HashMap::new();
    map.insert(
        "checkpoint.encoder.weight".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
        .with_name_mapping(|n| n.replace("model", "checkpoint"));

    // "model.encoder.weight" -> mapped to "checkpoint.encoder.weight" -> exists
    assert!(vb.pp("model").pp("encoder").contains_tensor("weight"));
    // "model.encoder.bias" -> mapped to "checkpoint.encoder.bias" -> doesn't exist
    assert!(!vb.pp("model").pp("encoder").contains_tensor("bias"));
}

#[test]
fn test_contains_tensor_with_hf_mapper() {
    let mut map = HashMap::new();
    map.insert(
        "model.vision_model.encoder.weight".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let mapper = HfToNnMapper::siglip2_granite_docling();
    let vb =
        VarBuilder::from_tensors(map, DType::F32, &Device::Cpu).with_weight_name_mapper(mapper);

    // NN name "encoder.weight" -> maps to "model.vision_model.encoder.weight"
    assert!(vb.pp("encoder").contains_tensor("weight"));
    assert!(!vb.pp("encoder").contains_tensor("bias"));
}

#[test]
fn test_contains_tensor_zeros_backend_always_true() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    assert!(vb.contains_tensor("anything"));
    assert!(vb.pp("deep").pp("path").contains_tensor("weight"));
    assert!(vb.contains_tensor("")); // even empty name
}

#[test]
fn test_contains_tensor_empty_map_always_false() {
    let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu);
    assert!(!vb.contains_tensor("weight"));
    assert!(!vb.pp("encoder").contains_tensor("weight"));
}

// ===========================================================================
// I. VarBuilder clone and sharing semantics
// ===========================================================================

#[test]
fn test_clone_shares_backend_arc() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    let vb1 = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let vb2 = vb1.clone();

    // Both see the same tensors.
    let t1 = vb1.get(&[2], "w").unwrap();
    let t2 = vb2.get(&[2], "w").unwrap();
    assert_eq!(
        t1.to_flat_vec::<f32>().unwrap(),
        t2.to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_clone_prefix_independence() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu).pp("root");
    let clone = vb.clone();
    let derived = clone.pp("child");

    assert_eq!(vb.prefix(), "root");
    assert_eq!(clone.prefix(), "root");
    assert_eq!(derived.prefix(), "root.child");
}

#[test]
fn test_clone_preserves_dtype_and_device() {
    let vb = VarBuilder::zeros(DType::BF16, &Device::metal());
    let clone = vb;
    assert_eq!(clone.dtype(), DType::BF16);
    assert_eq!(*clone.device(), Device::metal());
}

#[test]
fn test_clone_preserves_name_mapping() {
    let vb =
        VarBuilder::zeros(DType::F32, &Device::Cpu).with_name_mapping(|n| format!("prefix.{n}"));
    let clone = vb;
    assert!(clone.has_name_mapping());
}

#[test]
fn test_clone_preserves_precision_policy() {
    use crate::mixed_precision::MixedPrecisionPolicy;
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu).with_precision_policy(policy.clone());
    let clone = vb;
    assert_eq!(clone.precision_policy(), Some(&policy));
    assert_eq!(clone.effective_weight_dtype(), DType::BF16);
}

#[test]
fn test_pp_returns_independent_copy() {
    let mut map = HashMap::new();
    map.insert(
        "a.weight".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    map.insert(
        "b.weight".to_string(),
        DynTensor::new(&[2.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);

    let a = vb.pp("a");
    let b = vb.pp("b");

    // Each branch accesses its own scope.
    assert_eq!(
        a.get(&[1], "weight").unwrap().to_flat_vec::<f32>().unwrap(),
        vec![1.0]
    );
    assert_eq!(
        b.get(&[1], "weight").unwrap().to_flat_vec::<f32>().unwrap(),
        vec![2.0]
    );

    // Cross-scope access fails.
    assert!(!a.contains_tensor("b.weight"));
    assert!(!b.contains_tensor("a.weight"));
}

#[test]
fn test_shared_varbuilder_across_threads() {
    let mut map = HashMap::new();
    map.insert(
        "shared.w".to_string(),
        DynTensor::new(&[42.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = Arc::new(VarBuilder::from_tensors(map, DType::F32, &Device::Cpu));

    let handles: Vec<_> = (0..8)
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
        assert_eq!(result, vec![42.0]);
    }
}

#[test]
fn test_to_dtype_and_to_device_return_new_builder() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu).pp("model");
    let vb_bf16 = vb.to_dtype(DType::BF16);
    let vb_metal = vb.to_device(Device::metal());

    // Original unchanged.
    assert_eq!(vb.dtype(), DType::F32);
    assert_eq!(*vb.device(), Device::Cpu);
    assert_eq!(vb.prefix(), "model");

    // Derived builders have new dtype/device but same prefix.
    assert_eq!(vb_bf16.dtype(), DType::BF16);
    assert_eq!(vb_bf16.prefix(), "model");
    assert_eq!(*vb_metal.device(), Device::metal());
    assert_eq!(vb_metal.prefix(), "model");
}

#[test]
fn test_debug_format_includes_prefix_and_dtype() {
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu)
        .pp("encoder")
        .pp("layers")
        .pp("0");
    let debug = format!("{vb:?}");
    assert!(debug.contains("VarBuilder"));
    assert!(debug.contains("encoder.layers.0"));
    assert!(debug.contains("BF16"));
}

#[test]
fn test_as_ref_returns_self() {
    fn accepts_ref(vb: &VarBuilder) -> String {
        vb.prefix()
    }
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu).pp("test");
    // AsRef<VarBuilder> for VarBuilder returns self.
    let prefix = accepts_ref(vb.as_ref());
    assert_eq!(prefix, "test");
}
