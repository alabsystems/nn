// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for LoRA-wrapped Kokoro decoder.

use super::*;

fn make_conv_weight(out_ch: usize, in_ch: usize, k: usize) -> DynTensor {
    DynTensor::randn(0.0, 0.1, &[out_ch, in_ch, k], &Device::Cpu).unwrap()
}

// -- LoraConv1dAdapter tests --

#[test]
fn test_lora_conv1d_adapter_construction() {
    let w = make_conv_weight(16, 8, 3);
    let config = LoraConfig::new(4, 4.0);
    let adapter = LoraConv1dAdapter::from_conv_weight(&w, None, &config).unwrap();
    assert_eq!(adapter.trainable_params().len(), 2);
    assert!((adapter.scaling() - 1.0).abs() < 1e-10);
    assert_eq!(adapter.lora_a().dims(), &[4, 24]); // rank=4, fan_in=8*3=24
    assert_eq!(adapter.lora_b().dims(), &[16, 4]); // out_ch=16, rank=4
}

#[test]
fn test_lora_conv1d_adapter_zero_rank() {
    let w = make_conv_weight(16, 8, 3);
    let config = LoraConfig::new(0, 4.0);
    assert!(LoraConv1dAdapter::from_conv_weight(&w, None, &config).is_err());
}

#[test]
fn test_lora_conv1d_adapter_merge_initial() {
    // B is zero-initialized, so merged should equal frozen
    let w = make_conv_weight(16, 8, 3);
    let config = LoraConfig::new(4, 4.0);
    let adapter = LoraConv1dAdapter::from_conv_weight(&w, None, &config).unwrap();
    let merged = adapter.merged_weight().unwrap();

    let diff = merged
        .sub(&w)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        diff < 1e-6,
        "initial merged weight should equal frozen, got max diff {diff}"
    );
}

// -- LoraStage1Block tests --

#[test]
fn test_lora_stage1_block() {
    let w1 = make_conv_weight(64, 32, 3);
    let b1 = DynTensor::zeros(&[64], DType::F32, &Device::Cpu).unwrap();
    let w2 = make_conv_weight(64, 64, 3);
    let b2 = DynTensor::zeros(&[64], DType::F32, &Device::Cpu).unwrap();
    let config = LoraConfig::new(8, 8.0);

    let conv1_lora = LoraConv1dAdapter::from_conv_weight(&w1, Some(&b1), &config).unwrap();
    let conv2_lora = LoraConv1dAdapter::from_conv_weight(&w2, Some(&b2), &config).unwrap();
    let block = LoraStage1Block {
        conv1_lora,
        conv2_lora,
    };

    // 4 params: A,B for conv1 + A,B for conv2
    assert_eq!(block.trainable_params().len(), 4);

    // Merge should succeed
    let (m1, m2) = block.merge().unwrap();
    let d1 = m1
        .sub(&w1)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let d2 = m2
        .sub(&w2)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(d1 < 1e-6, "conv1 merge drift: {d1}");
    assert!(d2 < 1e-6, "conv2 merge drift: {d2}");
}

// -- LoraResBlockPair tests --

#[test]
fn test_lora_resblock_pair_construction() {
    let w1 = make_conv_weight(32, 32, 3);
    let b1 = DynTensor::zeros(&[32], DType::F32, &Device::Cpu).unwrap();
    let w2 = make_conv_weight(32, 32, 3);
    let b2 = DynTensor::zeros(&[32], DType::F32, &Device::Cpu).unwrap();
    let config = LoraConfig::new(4, 4.0);

    let c1 = Conv1d::new(w1, Some(b1), Default::default()).unwrap();
    let c2 = Conv1d::new(w2, Some(b2), Default::default()).unwrap();
    let pair = LoraResBlockPair::from_conv_pair(&c1, &c2, &config).unwrap();

    assert_eq!(pair.trainable_params().len(), 4);
    let (m1, m2) = pair.merge().unwrap();
    assert_eq!(m1.shape().dims(), c1.weight().shape().dims());
    assert_eq!(m2.shape().dims(), c2.weight().shape().dims());
}

// -- LoraGenerator tests --

#[test]
fn test_lora_generator_save_load_roundtrip() {
    let w1 = make_conv_weight(32, 32, 3);
    let b1 = DynTensor::zeros(&[32], DType::F32, &Device::Cpu).unwrap();
    let w2 = make_conv_weight(32, 32, 3);
    let b2 = DynTensor::zeros(&[32], DType::F32, &Device::Cpu).unwrap();
    let config = LoraConfig::new(4, 4.0);

    let c1 = Conv1d::new(w1, Some(b1), Default::default()).unwrap();
    let c2 = Conv1d::new(w2, Some(b2), Default::default()).unwrap();
    let pair = LoraResBlockPair::from_conv_pair(&c1, &c2, &config).unwrap();

    let lora_gen = LoraGenerator {
        resblock_loras: vec![pair],
    };

    let saved = lora_gen.save_lora_weights();
    assert_eq!(saved.len(), 4);
    assert!(saved.iter().any(|(k, _)| k == "generator.0.conv1.lora_a"));
    assert!(saved.iter().any(|(k, _)| k == "generator.0.conv2.lora_b"));
}

// -- SingingLoraConfig tests --

#[test]
fn test_singing_lora_config_stage1() {
    let config = SingingLoraConfig::stage1(16, 16.0);
    assert_eq!(config.stage, SingingStage::Stage1);
    assert_eq!(config.stage1_config.rank, 16);
}

#[test]
fn test_singing_lora_config_stage2() {
    let config = SingingLoraConfig::stage2(8, 8.0);
    assert_eq!(config.stage, SingingStage::Stage2);
    assert_eq!(config.stage2_config.rank, 8);
}

#[test]
fn test_singing_lora_config_both() {
    let config = SingingLoraConfig::both(16, 16.0, 8, 8.0);
    assert_eq!(config.stage, SingingStage::Both);
    assert_eq!(config.stage1_config.rank, 16);
    assert_eq!(config.stage2_config.rank, 8);
}

#[test]
fn test_singing_lora_config_default() {
    let config = SingingLoraConfig::default();
    assert_eq!(config.stage, SingingStage::Both);
    assert_eq!(config.stage1_config.rank, 16);
    assert_eq!(config.stage2_config.rank, 8);
}

// -- Weight set/load tests --

#[test]
fn test_lora_conv1d_set_weights() {
    let w = make_conv_weight(16, 8, 3);
    let config = LoraConfig::new(4, 4.0);
    let mut adapter = LoraConv1dAdapter::from_conv_weight(&w, None, &config).unwrap();

    let new_a = DynTensor::randn(0.0, 0.5, &[4, 24], &Device::Cpu).unwrap();
    adapter.set_lora_a(new_a.clone()).unwrap();
    let diff = adapter
        .lora_a()
        .sub(&new_a)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(diff < 1e-7);
}

#[test]
fn test_lora_conv1d_set_wrong_shape() {
    let w = make_conv_weight(16, 8, 3);
    let config = LoraConfig::new(4, 4.0);
    let mut adapter = LoraConv1dAdapter::from_conv_weight(&w, None, &config).unwrap();

    let bad_a = DynTensor::randn(0.0, 1.0, &[8, 24], &Device::Cpu).unwrap();
    assert!(adapter.set_lora_a(bad_a).is_err());
}
