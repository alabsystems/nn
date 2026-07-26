// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! VarBuilder loading, position interpolation, and Debug impl tests for ViT.
//!
//! Extracted from `vit_tests.rs` for 500-line compliance (#1306).

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::Device;

// -- Position embedding interpolation -----------------------------------------

#[test]
fn test_vit_encoder_position_interpolation() {
    // Build encoder with image_size=8 (4 patches), run on 16x16 (16 patches)
    let config = small_config(false);
    let encoder = make_encoder(&config, 1);
    // Larger image: 16x16 with patch_size=4 -> 16 patches (triggers interpolation)
    let larger_img =
        DynTensor::from_vec(det_data(3 * 16 * 16, 0.0), &[1, 3, 16, 16], &Device::Cpu).unwrap();
    let out = encoder.forward(&larger_img, PoolingStrategy::None).unwrap();
    assert_eq!(out.dims(), &[1, 16, 32]);
    assert!(out.as_cpu_f32().unwrap().iter().all(|v| v.is_finite()));
}

#[test]
fn test_vit_encoder_position_interpolation_with_cls() {
    // CLS-aware interpolation path: separates CLS pos, interpolates patch pos, re-cats
    let config = small_config(true);
    let encoder = make_encoder(&config, 1);
    // 16x16 with patch_size=4 -> 16 patches + 1 CLS = 17 seq_len (triggers interpolation)
    let larger_img =
        DynTensor::from_vec(det_data(3 * 16 * 16, 0.0), &[1, 3, 16, 16], &Device::Cpu).unwrap();
    let out = encoder.forward(&larger_img, PoolingStrategy::None).unwrap();
    assert_eq!(out.dims(), &[1, 17, 32]);
    assert!(out.as_cpu_f32().unwrap().iter().all(|v| v.is_finite()));
    // CLS pooling should still work on interpolated sequence
    let cls_out = encoder.forward(&larger_img, PoolingStrategy::Cls).unwrap();
    assert_eq!(cls_out.dims(), &[1, 32]);
}

// -- Debug impls --------------------------------------------------------------

#[test]
fn test_debug_impls() {
    let config = small_config(false);
    let pe = make_patch_embed(&config);
    let debug = format!("{pe:?}");
    assert!(debug.contains("PatchEmbedding"), "debug: {debug}");

    let block = make_encoder_block(32, 4, 64);
    let debug = format!("{block:?}");
    assert!(debug.contains("VitEncoderBlock"), "debug: {debug}");

    let encoder = make_encoder(&config, 2);
    let debug = format!("{encoder:?}");
    assert!(debug.contains("VitEncoder"), "debug: {debug}");
    assert!(debug.contains("num_layers"), "debug: {debug}");
}

// -- VarBuilder load tests ----------------------------------------------------

#[test]
fn test_vit_encoder_load_from_zeros_backend() {
    use crate::var_builder::VarBuilder;
    use crate::DType;

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let config = small_config(true);
    let encoder = VitEncoder::load(&vb, &config).expect("load from ZerosBackend should succeed");
    assert_eq!(encoder.blocks().len(), config.num_layers);

    // Forward pass with zero weights should produce finite output
    let img = make_image(1, &config, 0.0);
    let out = encoder.forward(&img, PoolingStrategy::None).unwrap();
    assert_eq!(out.dims(), &[1, config.seq_len(), config.hidden_size]);
    assert!(out.as_cpu_f32().unwrap().iter().all(|v| v.is_finite()));
}

#[test]
fn test_vit_encoder_load_no_cls() {
    use crate::var_builder::VarBuilder;
    use crate::DType;

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let config = small_config(false);
    let encoder = VitEncoder::load(&vb, &config).expect("load without CLS should succeed");
    assert_eq!(encoder.blocks().len(), config.num_layers);

    let img = make_image(1, &config, 0.0);
    let out = encoder.forward(&img, PoolingStrategy::Mean).unwrap();
    assert_eq!(out.dims(), &[1, config.hidden_size]);
}

#[test]
fn test_vit_encoder_block_load_from_zeros_backend() {
    use crate::var_builder::VarBuilder;
    use crate::DType;

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let block = VitEncoderBlock::load(&vb, 32, 4, 64, 1e-6)
        .expect("block load from ZerosBackend should succeed");

    let x = DynTensor::from_vec(det_data(2 * 5 * 32, 0.0), &[2, 5, 32], &Device::Cpu).unwrap();
    let out = block.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 5, 32]);
}

#[test]
fn test_patch_embedding_load_from_zeros_backend() {
    use crate::var_builder::VarBuilder;
    use crate::DType;

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let config = small_config(false);
    let pe = PatchEmbedding::load(&vb, &config)
        .expect("PatchEmbedding load from ZerosBackend should succeed");

    let img = make_image(1, &config, 0.0);
    let out = pe.forward(&img).unwrap();
    assert_eq!(out.dims(), &[1, config.num_patches(), config.hidden_size]);
}
