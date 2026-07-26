// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional VarBuilder tests covering construction, prefix navigation,
//! weight lookup, error handling, and dtype/device propagation.
//!
//! These tests complement the per-module tests in `var_builder/tests.rs`.

use std::collections::HashMap;
use std::sync::Arc;

use crate::dyn_tensor::DynTensor;
use crate::var_builder::{TensorBackend, VarBuilder};
use crate::{DType, Device, TensorError};

// ---------------------------------------------------------------------------
// A. VarBuilder::zeros construction
// ---------------------------------------------------------------------------

#[test]
fn test_zeros_f32_cpu_produces_correct_dtype_and_device() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    assert_eq!(vb.dtype(), DType::F32);
    assert_eq!(*vb.device(), Device::Cpu);
    assert_eq!(vb.prefix(), "");
}

#[test]
fn test_zeros_bf16_cpu_tensor_dtype() {
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let t = vb.get(&[4], "w").unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    assert_eq!(t.dims(), &[4]);
}

#[test]
fn test_zeros_f16_produces_f16_tensors() {
    let vb = VarBuilder::zeros(DType::F16, &Device::Cpu);
    let t = vb.get(&[2, 3], "x").unwrap();
    assert_eq!(t.dtype(), DType::F16);
}

#[test]
fn test_zeros_metal_device_propagates() {
    let vb = VarBuilder::zeros(DType::F32, &Device::metal());
    assert_eq!(*vb.device(), Device::Metal { device_id: 0 });
}

#[test]
fn test_zeros_empty_shape_returns_scalar() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let t = vb.get(&[], "s").unwrap();
    assert_eq!(t.dims(), &[] as &[usize]);
}

#[test]
fn test_zeros_1d_shape() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let t = vb.get(&[10], "vec").unwrap();
    assert_eq!(t.dims(), &[10]);
    let data = t.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| v == 0.0));
    assert_eq!(data.len(), 10);
}

#[test]
fn test_zeros_high_rank_shape() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let t = vb.get(&[2, 3, 4, 5], "high_rank").unwrap();
    assert_eq!(t.dims(), &[2, 3, 4, 5]);
}

// ---------------------------------------------------------------------------
// B. VarBuilder::from_tensors with HashMap
// ---------------------------------------------------------------------------

#[test]
fn test_from_tensors_empty_map() {
    let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu);
    assert_eq!(vb.dtype(), DType::F32);
    assert!(vb.tensor_names().is_empty());
}

#[test]
fn test_from_tensors_single_entry() {
    let mut map = HashMap::new();
    map.insert(
        "bias".to_string(),
        DynTensor::new(&[0.5], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    assert!(vb.contains_tensor("bias"));
    assert!(!vb.contains_tensor("weight"));
    let t = vb.get(&[1], "bias").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![0.5]);
}

#[test]
fn test_from_tensors_multiple_entries_accessible() {
    let mut map = HashMap::new();
    map.insert(
        "a.w".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    map.insert(
        "b.w".to_string(),
        DynTensor::new(&[3.0, 4.0], &[2], &Device::Cpu).unwrap(),
    );
    map.insert(
        "c.w".to_string(),
        DynTensor::new(&[5.0, 6.0], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    assert_eq!(
        vb.pp("a")
            .get(&[2], "w")
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        vb.pp("b")
            .get(&[2], "w")
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap(),
        vec![3.0, 4.0]
    );
    assert_eq!(
        vb.pp("c")
            .get(&[2], "w")
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap(),
        vec![5.0, 6.0]
    );
}

// ---------------------------------------------------------------------------
// C. Prefix path navigation (pp)
// ---------------------------------------------------------------------------

#[test]
fn test_pp_single_level() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let child = vb.pp("decoder");
    assert_eq!(child.prefix(), "decoder");
}

#[test]
fn test_pp_nested_encoder_layer() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let deep = vb.pp("encoder").pp("layer.0");
    assert_eq!(deep.prefix(), "encoder.layer.0");
}

#[test]
fn test_pp_deeply_nested_five_levels() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let deep = vb.pp("model").pp("encoder").pp("layers").pp("0").pp("attn");
    assert_eq!(deep.prefix(), "model.encoder.layers.0.attn");
}

#[test]
fn test_pp_empty_string_skipped_at_start() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let child = vb.pp("").pp("encoder");
    assert_eq!(child.prefix(), "encoder");
}

#[test]
fn test_pp_empty_string_skipped_in_middle() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let child = vb.pp("a").pp("").pp("b");
    assert_eq!(child.prefix(), "a.b");
}

#[test]
fn test_pp_empty_string_skipped_at_end() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let child = vb.pp("encoder").pp("");
    assert_eq!(child.prefix(), "encoder");
}

#[test]
fn test_pp_multiple_empty_strings_all_skipped() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let child = vb.pp("").pp("").pp("x").pp("").pp("");
    assert_eq!(child.prefix(), "x");
}

#[test]
fn test_pp_does_not_modify_parent_prefix() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let _child = vb.pp("child").pp("grandchild");
    assert_eq!(vb.prefix(), "");
}

#[test]
fn test_pp_numeric_string_segments() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let child = vb.pp("layers").pp("42");
    assert_eq!(child.prefix(), "layers.42");
}

// ---------------------------------------------------------------------------
// D. Weight lookup (get, get_unchecked)
// ---------------------------------------------------------------------------

#[test]
fn test_get_with_prefix_resolves_full_key() {
    let mut map = HashMap::new();
    map.insert(
        "model.encoder.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let t = vb.pp("model").pp("encoder").get(&[3], "weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_get_unchecked_returns_stored_shape() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let t = vb.get_unchecked("w").unwrap();
    assert_eq!(t.dims(), &[2, 3]);
}

#[test]
fn test_get_validates_shape() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let err = vb.get(&[3], "w").unwrap_err();
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
fn test_contains_tensor_with_prefix() {
    let mut map = HashMap::new();
    map.insert(
        "encoder.weight".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    assert!(vb.pp("encoder").contains_tensor("weight"));
    assert!(!vb.pp("encoder").contains_tensor("bias"));
    assert!(!vb.pp("decoder").contains_tensor("weight"));
}

// ---------------------------------------------------------------------------
// E. Missing weight error handling
// ---------------------------------------------------------------------------

#[test]
fn test_missing_weight_error_includes_full_path() {
    let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu);
    let err = vb
        .pp("encoder")
        .pp("layer.0")
        .get(&[4], "weight")
        .unwrap_err();
    match err {
        TensorError::TensorNotFound { name } => {
            assert_eq!(name, "encoder.layer.0.weight");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

#[test]
fn test_missing_weight_error_no_prefix() {
    let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu);
    let err = vb.get(&[1], "nonexistent").unwrap_err();
    match err {
        TensorError::TensorNotFound { name } => {
            assert_eq!(name, "nonexistent");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

#[test]
fn test_missing_weight_get_unchecked_error() {
    let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu);
    let err = vb.pp("a").get_unchecked("b").unwrap_err();
    match err {
        TensorError::TensorNotFound { name } => {
            assert_eq!(name, "a.b");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// F. DType/Device propagation through VarBuilder
// ---------------------------------------------------------------------------

#[test]
fn test_dtype_propagates_through_pp_chain() {
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let child = vb.pp("a").pp("b").pp("c");
    assert_eq!(child.dtype(), DType::BF16);
}

#[test]
fn test_device_propagates_through_pp_chain() {
    let device = Device::Cuda { device_id: 2 };
    let vb = VarBuilder::zeros(DType::F32, &device);
    let child = vb.pp("model").pp("layers");
    assert_eq!(*child.device(), Device::Cuda { device_id: 2 });
}

#[test]
fn test_to_dtype_returns_new_builder() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let vb2 = vb.to_dtype(DType::F16);
    assert_eq!(vb.dtype(), DType::F32);
    assert_eq!(vb2.dtype(), DType::F16);
}

#[test]
fn test_to_device_returns_new_builder() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let vb2 = vb.to_device(Device::Cuda { device_id: 0 });
    assert_eq!(*vb.device(), Device::Cpu);
    assert_eq!(*vb2.device(), Device::Cuda { device_id: 0 });
}

#[test]
fn test_to_dtype_and_to_device_chained() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let vb2 = vb.to_dtype(DType::BF16).to_device(Device::metal());
    assert_eq!(vb2.dtype(), DType::BF16);
    assert_eq!(*vb2.device(), Device::metal());
    // Original unchanged.
    assert_eq!(vb.dtype(), DType::F32);
    assert_eq!(*vb.device(), Device::Cpu);
}

#[test]
fn test_to_dtype_preserves_prefix() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu).pp("encoder");
    let vb2 = vb.to_dtype(DType::BF16);
    assert_eq!(vb2.prefix(), "encoder");
    assert_eq!(vb2.dtype(), DType::BF16);
}

#[test]
fn test_to_device_preserves_prefix() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu).pp("model");
    let vb2 = vb.to_device(Device::vulkan());
    assert_eq!(vb2.prefix(), "model");
    assert_eq!(*vb2.device(), Device::vulkan());
}

// ---------------------------------------------------------------------------
// G. from_backend custom backend
// ---------------------------------------------------------------------------

#[test]
fn test_from_backend_custom_implementation() {
    /// A backend that always returns ones.
    struct OnesBackend;
    impl TensorBackend for OnesBackend {
        fn get(
            &self,
            dims: &[usize],
            _name: &str,
            dtype: DType,
            device: &Device,
        ) -> crate::Result<DynTensor> {
            DynTensor::ones(dims, dtype, device)
        }
        fn get_unchecked(
            &self,
            _name: &str,
            dtype: DType,
            device: &Device,
        ) -> crate::Result<DynTensor> {
            DynTensor::ones(&[1], dtype, device)
        }
        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    let vb = VarBuilder::from_backend(Arc::new(OnesBackend), DType::F32, Device::Cpu);
    let t = vb.get(&[3], "anything").unwrap();
    let data = t.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![1.0, 1.0, 1.0]);
}

// ---------------------------------------------------------------------------
// H. Nested prefix paths (pp("encoder").pp("layer.0"))
// ---------------------------------------------------------------------------

#[test]
fn test_nested_prefix_with_dot_in_segment() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let child = vb.pp("encoder").pp("layer.0");
    assert_eq!(child.prefix(), "encoder.layer.0");
}

#[test]
fn test_nested_prefix_resolves_tensor_key() {
    let mut map = HashMap::new();
    map.insert(
        "encoder.layer.0.weight".to_string(),
        DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap(),
    );
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let t = vb.pp("encoder").pp("layer.0").get(&[2], "weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn test_nested_prefix_missing_returns_correct_error_path() {
    let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu);
    let err = vb
        .pp("encoder")
        .pp("layer.0")
        .pp("attn")
        .get(&[1], "q_proj.weight")
        .unwrap_err();
    match err {
        TensorError::TensorNotFound { name } => {
            assert_eq!(name, "encoder.layer.0.attn.q_proj.weight");
        }
        other => panic!("expected TensorNotFound, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// I. VarBuilder clone and Debug
// ---------------------------------------------------------------------------

#[test]
fn test_clone_preserves_all_properties() {
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu)
        .pp("model")
        .to_device(Device::metal());
    let cloned = vb.clone();
    assert_eq!(cloned.prefix(), vb.prefix());
    assert_eq!(cloned.dtype(), vb.dtype());
    assert_eq!(*cloned.device(), *vb.device());
}

#[test]
fn test_debug_shows_prefix_and_dtype() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu)
        .pp("encoder")
        .pp("layer");
    let debug = format!("{vb:?}");
    assert!(debug.contains("VarBuilder"));
    assert!(debug.contains("encoder.layer"));
    assert!(debug.contains("F32"));
}

// ---------------------------------------------------------------------------
// J. Precision policy interaction
// ---------------------------------------------------------------------------

#[test]
fn test_precision_policy_not_set_by_default() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    assert!(vb.precision_policy().is_none());
    assert_eq!(vb.effective_weight_dtype(), DType::F32);
}

// ---------------------------------------------------------------------------
// K. Name mapping through pp
// ---------------------------------------------------------------------------

#[test]
fn test_has_name_mapping_false_by_default() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    assert!(!vb.has_name_mapping());
}

#[test]
fn test_has_name_mapping_true_after_with_name_mapping() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu).with_name_mapping(ToString::to_string);
    assert!(vb.has_name_mapping());
}

#[test]
fn test_name_mapping_propagates_through_pp() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu).with_name_mapping(ToString::to_string);
    let child = vb.pp("a").pp("b");
    assert!(child.has_name_mapping());
}

// ---------------------------------------------------------------------------
// L. AsRef<VarBuilder> implementation
// ---------------------------------------------------------------------------

#[test]
fn test_as_ref_returns_self() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu).pp("test");
    let vb_ref: &VarBuilder = vb.as_ref();
    assert_eq!(vb_ref.prefix(), "test");
}

// ---------------------------------------------------------------------------
// M. Send + Sync bounds
// ---------------------------------------------------------------------------

#[test]
fn test_varbuilder_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<VarBuilder>();
}
