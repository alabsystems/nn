#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Model forward pass and BF16 dtype conversion tests extracted from
//! `glm5_tests.rs` for 500-line compliance.

use super::*;
use crate::test_utils::tiny_config;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// -- Model load & forward -----------------------------------------------------

#[test]
fn test_model_load_zeros() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg);
    assert!(model.is_ok(), "model load failed: {:?}", model.err());
}

#[test]
fn test_model_forward_zeros() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let input_ids = &[0, 1, 2];
    let positions = &[0, 1, 2];
    let result = model.forward(input_ids, positions);
    assert!(
        result.is_ok(),
        "zero-weight forward should succeed: {:?}",
        result.err()
    );
    let output = result.unwrap();
    let vals = output.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "all output values should be finite with zero weights"
    );
}

#[test]
fn test_model_forward_single_token() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let logits = model.forward(&[42], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, 100]); // [1, 1, padded_vocab_size]
}

#[test]
fn test_model_forward_output_shape() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let logits = model.forward(&[0, 1, 2, 3], &[0, 1, 2, 3]).unwrap();
    assert_eq!(logits.dims(), &[1, 4, 100]); // [1, seq_len, padded_vocab_size]
}

#[test]
fn test_model_forward_mismatched_lengths() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let err = model.forward(&[0, 1], &[0]);
    assert!(err.is_err());
}

#[test]
fn test_model_forward_from_embeddings() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let embeddings = DynTensor::zeros(&[1, 3, 256], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&embeddings, &[0, 1, 2], None);
    assert!(
        result.is_ok(),
        "forward_from_embeddings failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().dims(), &[1, 3, 100]);
}

#[test]
fn test_model_forward_from_embeddings_wrong_hidden_size() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let embeddings = DynTensor::zeros(&[1, 3, 128], DType::F32, &Device::Cpu).unwrap();
    let err = model.forward_from_embeddings(&embeddings, &[0, 1, 2], None);
    assert!(err.is_err());
}

#[test]
fn test_model_no_bias_linear() {
    // Default tiny_config has add_bias_linear=false
    let cfg = tiny_config();
    assert!(!cfg.add_bias_linear);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg);
    assert!(model.is_ok());
}

#[test]
fn test_model_with_bias_linear() {
    let mut cfg = tiny_config();
    cfg.add_bias_linear = true;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg);
    assert!(
        model.is_ok(),
        "model with bias should load: {:?}",
        model.err()
    );
}

#[test]
fn test_model_no_qkv_bias() {
    let mut cfg = tiny_config();
    cfg.add_qkv_bias = false;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg);
    assert!(
        model.is_ok(),
        "model without QKV bias should load: {:?}",
        model.err()
    );
}

#[test]
fn test_model_new_cache() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let cache = model.new_cache();
    assert_eq!(cache.num_layers(), 2);
}

#[test]
fn test_model_cached_forward() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    // First token
    let logits1 = model.forward_cached(&[0], &[0], Some(&mut cache)).unwrap();
    assert_eq!(logits1.dims(), &[1, 1, 100]);
    assert_eq!(cache.seq_len(), 1);

    // Second token
    let logits2 = model.forward_cached(&[1], &[1], Some(&mut cache)).unwrap();
    assert_eq!(logits2.dims(), &[1, 1, 100]);
    assert_eq!(cache.seq_len(), 2);
}

#[test]
fn test_model_cache_wrong_layers() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let mut cache = KvCache::new(5); // wrong: model has 2 layers

    let err = model.forward_cached(&[0], &[0], Some(&mut cache));
    assert!(err.is_err());
}

#[test]
fn test_nan_input_rejected() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let mut data = vec![0.0f32; 2 * 256];
    data[0] = f32::NAN;
    let nan_embeddings = DynTensor::from_vec(data, &[1, 2, 256], &Device::Cpu).unwrap();

    let err = model.forward_from_embeddings(&nan_embeddings, &[0, 1], None);
    assert!(err.is_err(), "NaN input should be rejected");
}

// -- BF16 dtype conversion tests (#1734) --------------------------------------

#[test]
fn test_forward_from_embeddings_bf16_model_f32_input() {
    // BF16 model with F32 embeddings — should auto-convert (#1734).
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    assert_eq!(model.dtype(), DType::BF16);

    let embeddings = DynTensor::ones(&[1, 2, 256], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&embeddings, &[0, 1], None);
    assert!(
        result.is_ok(),
        "bf16 model with f32 embeddings should succeed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().dims(), &[1, 2, 100]);
}

#[test]
fn test_forward_from_embeddings_bf16_model_bf16_input() {
    // BF16 model with BF16 embeddings — should work without conversion.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let embeddings = DynTensor::zeros(&[1, 2, 256], DType::BF16, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&embeddings, &[0, 1], None);
    assert!(
        result.is_ok(),
        "bf16 model with bf16 embeddings should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_forward_from_embeddings_f32_model_unchanged() {
    // F32 model — embeddings are not converted (regression test for #1734).
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    assert_eq!(model.dtype(), DType::F32);

    let embeddings = DynTensor::ones(&[1, 2, 256], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&embeddings, &[0, 1], None);
    assert!(
        result.is_ok(),
        "f32 model with f32 embeddings should still work: {:?}",
        result.err()
    );
}
