// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SigLIP2 Vision Encoder.

use super::{SigLip2Config, SigLip2VisionEncoder};
use crate::layers::vision::PoolingStrategy;
use crate::var_builder::VarBuilder;
use crate::{DType, Device};

/// Tiny SigLIP2 config for tests that need a forward pass but don't test
/// production-scale behavior. hidden=32, 2 layers, 2 heads, patch=16.
/// Full base_patch16 (768, 12 layers) takes >60s per forward pass on CPU.
fn tiny_patch16(image_size: usize) -> SigLip2Config {
    SigLip2Config::new(3, 32, 2, 2, 64, 16, image_size, 1e-6).unwrap()
}

#[test]
fn test_siglip2_base_config() {
    let config = SigLip2Config::base_patch16(224).unwrap();
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_layers, 12);
    assert_eq!(config.num_heads, 12);
    assert_eq!(config.patch_size, 16);
}

#[test]
fn test_siglip2_config_validation() {
    // patch_size=0 should fail
    let err = SigLip2Config::new(3, 768, 12, 12, 3072, 0, 224, 1e-6).unwrap_err();
    assert!(format!("{err:?}").contains("patch_size"));
}

#[test]
fn test_siglip2_load_and_forward_zeros() {
    // Use tiny config: full base_patch16 (768, 12 layers) takes >60s on CPU.
    let config = tiny_patch16(224);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = SigLip2VisionEncoder::load(&vb, &config).unwrap();

    let input =
        crate::dyn_tensor::DynTensor::zeros(&[1, 3, 224, 224], DType::F32, &Device::Cpu).unwrap();
    let output = encoder.forward(&input, PoolingStrategy::None).unwrap();
    // 224/16 = 14, 14*14 = 196 patches, hidden=32
    assert_eq!(output.dims(), &[1, 196, 32]);
}

#[test]
fn test_siglip2_mean_pooling() {
    let config = tiny_patch16(224);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = SigLip2VisionEncoder::load(&vb, &config).unwrap();

    let input =
        crate::dyn_tensor::DynTensor::zeros(&[2, 3, 224, 224], DType::F32, &Device::Cpu).unwrap();
    let output = encoder.forward(&input, PoolingStrategy::Mean).unwrap();
    assert_eq!(output.dims(), &[2, 32]);
}

#[test]
fn test_siglip2_cls_pooling_errors() {
    let config = tiny_patch16(224);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = SigLip2VisionEncoder::load(&vb, &config).unwrap();

    let input =
        crate::dyn_tensor::DynTensor::zeros(&[1, 3, 224, 224], DType::F32, &Device::Cpu).unwrap();
    let err = encoder.forward(&input, PoolingStrategy::Cls).unwrap_err();
    assert!(format!("{err:?}").contains("Cls pooling not supported"));
}

#[test]
fn test_siglip2_module_trait() {
    use crate::layers::Module;
    let config = tiny_patch16(224);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = SigLip2VisionEncoder::load(&vb, &config).unwrap();

    let input =
        crate::dyn_tensor::DynTensor::zeros(&[1, 3, 224, 224], DType::F32, &Device::Cpu).unwrap();
    let output = Module::forward(&encoder, &input).unwrap();
    assert_eq!(output.dims(), &[1, 196, 32]);
}

/// Build a tensor map with all SigLIP2 HuggingFace weight keys for testing.
/// Returns (config, tensors) for a small model: hidden=32, 2 layers, 2 heads.
fn build_siglip2_test_tensors() -> (
    SigLip2Config,
    std::collections::HashMap<String, crate::dyn_tensor::DynTensor>,
) {
    use crate::dyn_tensor::DynTensor;

    let d = 32_usize;
    let inter = 64_usize;
    let (ch, ps, img, n_layers) = (3, 4, 8, 2);
    let num_patches = (img / ps) * (img / ps);

    let config = SigLip2Config::new(ch, d, n_layers, 2, inter, ps, img, 1e-6).unwrap();
    let ones = |s: &[usize]| DynTensor::ones(s, DType::F32, &Device::Cpu).unwrap();

    let mut t = std::collections::HashMap::new();
    t.insert(
        "embeddings.patch_embedding.weight".into(),
        ones(&[d, ch, ps, ps]),
    );
    t.insert("embeddings.patch_embedding.bias".into(), ones(&[d]));
    t.insert(
        "embeddings.position_embedding.weight".into(),
        ones(&[num_patches, d]),
    );

    for i in 0..n_layers {
        let pfx = format!("encoder.layers.{i}");
        for ln in &["layer_norm1", "layer_norm2"] {
            t.insert(format!("{pfx}.{ln}.weight"), ones(&[d]));
            t.insert(format!("{pfx}.{ln}.bias"), ones(&[d]));
        }
        for proj in &["q_proj", "k_proj", "v_proj", "out_proj"] {
            t.insert(format!("{pfx}.self_attn.{proj}.weight"), ones(&[d, d]));
            t.insert(format!("{pfx}.self_attn.{proj}.bias"), ones(&[d]));
        }
        t.insert(format!("{pfx}.mlp.fc1.weight"), ones(&[inter, d]));
        t.insert(format!("{pfx}.mlp.fc1.bias"), ones(&[inter]));
        t.insert(format!("{pfx}.mlp.fc2.weight"), ones(&[d, inter]));
        t.insert(format!("{pfx}.mlp.fc2.bias"), ones(&[d]));
    }

    t.insert("post_layernorm.weight".into(), ones(&[d]));
    t.insert("post_layernorm.bias".into(), ones(&[d]));

    assert_eq!(t.len(), 5 + n_layers * 16); // 5 global + 16 per layer
    (config, t)
}

/// Verify SigLIP2 loads from a VarBuilder with explicit HuggingFace weight
/// key names (simulates safetensors checkpoint loading). Validates every
/// dotted key path resolves correctly through VarBuilder prefix scoping.
#[test]
fn test_siglip2_load_from_tensor_map() {
    let (config, tensors) = build_siglip2_test_tensors();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let encoder = SigLip2VisionEncoder::load(&vb, &config).unwrap();

    // Forward pass: [1, 3, 8, 8] -> [1, 4, 32]  (img=8, ps=4 -> 4 patches, d=32)
    let input =
        crate::dyn_tensor::DynTensor::ones(&[1, 3, 8, 8], DType::F32, &Device::Cpu).unwrap();
    let output = encoder.forward(&input, PoolingStrategy::None).unwrap();
    assert_eq!(output.dims(), &[1, 4, 32]);
}

/// Verify VarBuilder::from_tensors rejects missing weights (negative test).
#[test]
fn test_siglip2_load_missing_weight_errors() {
    use std::collections::HashMap;

    let config = SigLip2Config::new(3, 32, 1, 2, 64, 4, 8, 1e-6).unwrap();
    let tensors: HashMap<String, crate::dyn_tensor::DynTensor> = HashMap::new();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    // Loading with empty tensor map must fail (missing patch_embedding.weight)
    let err = SigLip2VisionEncoder::load(&vb, &config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("patch_embedding") || msg.contains("not found"),
        "Expected missing-tensor error, got: {msg}"
    );
}

// -- forward_deepstack --------------------------------------------------------

#[test]
fn test_siglip2_forward_deepstack_shape() {
    let (config, tensors) = build_siglip2_test_tensors();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let encoder = SigLip2VisionEncoder::load(&vb, &config).unwrap();

    let input =
        crate::dyn_tensor::DynTensor::ones(&[1, 3, 8, 8], DType::F32, &Device::Cpu).unwrap();
    // 2 layers: request both
    let outputs = encoder.forward_deepstack(&input, &[0, 1]).unwrap();
    assert_eq!(outputs.len(), 2);
    for out in &outputs {
        assert_eq!(out.dims(), &[1, 4, 32]); // [B, num_patches, D]
    }
}

#[test]
fn test_siglip2_forward_deepstack_empty_indices() {
    let (config, tensors) = build_siglip2_test_tensors();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let encoder = SigLip2VisionEncoder::load(&vb, &config).unwrap();

    let input =
        crate::dyn_tensor::DynTensor::ones(&[1, 3, 8, 8], DType::F32, &Device::Cpu).unwrap();
    let err = encoder
        .forward_deepstack(&input, &[])
        .unwrap_err()
        .to_string();
    assert!(err.contains("empty"), "error: {err}");
}

#[test]
fn test_siglip2_forward_deepstack_index_out_of_range() {
    let (config, tensors) = build_siglip2_test_tensors();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let encoder = SigLip2VisionEncoder::load(&vb, &config).unwrap();

    let input =
        crate::dyn_tensor::DynTensor::ones(&[1, 3, 8, 8], DType::F32, &Device::Cpu).unwrap();
    // 2 layers (indices 0,1), requesting index 5
    let err = encoder
        .forward_deepstack(&input, &[0, 5])
        .unwrap_err()
        .to_string();
    assert!(err.contains("out of range"), "error: {err}");
}

#[test]
fn test_siglip2_forward_deepstack_order_preserved() {
    let (config, tensors) = build_siglip2_test_tensors();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let encoder = SigLip2VisionEncoder::load(&vb, &config).unwrap();

    let input =
        crate::dyn_tensor::DynTensor::ones(&[1, 3, 8, 8], DType::F32, &Device::Cpu).unwrap();
    let fwd = encoder.forward_deepstack(&input, &[0, 1]).unwrap();
    let rev = encoder.forward_deepstack(&input, &[1, 0]).unwrap();

    // rev[0] (layer 1) should equal fwd[1] (layer 1)
    let rev_0: Vec<f32> = rev[0].to_flat_vec::<f32>().unwrap();
    let fwd_1: Vec<f32> = fwd[1].to_flat_vec::<f32>().unwrap();
    assert_eq!(rev_0.len(), fwd_1.len());
    for (a, b) in rev_0.iter().zip(fwd_1.iter()) {
        assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
    }
}

/// Verify 512x512 image path produces correct patch count.
/// Uses tiny config: full base_patch16 with 512 input takes >120s on CPU.
#[test]
fn test_siglip2_512_image_size() {
    let config = tiny_patch16(512);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = SigLip2VisionEncoder::load(&vb, &config).unwrap();

    let input =
        crate::dyn_tensor::DynTensor::zeros(&[1, 3, 512, 512], DType::F32, &Device::Cpu).unwrap();
    let output = encoder.forward(&input, PoolingStrategy::None).unwrap();
    // 512/16 = 32, 32*32 = 1024 patches, hidden=32
    assert_eq!(output.dims(), &[1, 1024, 32]);
}
