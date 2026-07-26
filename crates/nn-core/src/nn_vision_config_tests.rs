// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for vision model configuration structs: VitConfig, SigLip2Config,
//! Qwen2VLVitConfig, and PoolingStrategy.

use crate::layers::{PoolingStrategy, Qwen2VLVitConfig, Qwen3VLVitConfig, SigLip2Config, VitConfig};

// ---------------------------------------------------------------------------
// VitConfig validation
// ---------------------------------------------------------------------------

#[test]
fn test_vit_config_validation_valid() {
    // ViT-B/16 style: hidden=768, 12 layers, 12 heads, intermediate=3072,
    // patch=16, image=224, eps=1e-6, with CLS token.
    let cfg = VitConfig::new(3, 768, 12, 12, 3072, 16, 224, 1e-6, true);
    assert!(cfg.is_ok(), "valid ViT-B/16 config should succeed");
}

#[test]
fn test_vit_config_validation_zero_patch() {
    let err = VitConfig::new(3, 768, 12, 12, 3072, 0, 224, 1e-6, true);
    assert!(err.is_err(), "patch_size=0 must be rejected");
}

#[test]
fn test_vit_config_validation_zero_image() {
    let err = VitConfig::new(3, 768, 12, 12, 3072, 16, 0, 1e-6, true);
    assert!(err.is_err(), "image_size=0 must be rejected");
}

#[test]
fn test_vit_config_validation_indivisible_image() {
    // 225 is not divisible by 16
    let err = VitConfig::new(3, 768, 12, 12, 3072, 16, 225, 1e-6, true);
    assert!(
        err.is_err(),
        "image_size not divisible by patch_size must fail"
    );
}

#[test]
fn test_vit_config_validation_zero_hidden() {
    let err = VitConfig::new(3, 0, 12, 12, 3072, 16, 224, 1e-6, true);
    assert!(err.is_err(), "hidden_size=0 must be rejected");
}

#[test]
fn test_vit_config_validation_zero_channels() {
    let err = VitConfig::new(0, 768, 12, 12, 3072, 16, 224, 1e-6, true);
    assert!(err.is_err(), "num_channels=0 must be rejected");
}

#[test]
fn test_vit_config_validation_zero_intermediate() {
    let err = VitConfig::new(3, 768, 12, 12, 0, 16, 224, 1e-6, true);
    assert!(err.is_err(), "intermediate_size=0 must be rejected");
}

#[test]
fn test_vit_config_validation_indivisible_hidden_heads() {
    // hidden_size=768 is not divisible by num_heads=7
    let err = VitConfig::new(3, 768, 12, 7, 3072, 16, 224, 1e-6, true);
    assert!(
        err.is_err(),
        "hidden_size not divisible by num_heads must fail"
    );
}

#[test]
fn test_vit_config_validation_zero_heads() {
    let err = VitConfig::new(3, 768, 12, 0, 3072, 16, 224, 1e-6, true);
    assert!(err.is_err(), "num_heads=0 must be rejected");
}

#[test]
fn test_vit_config_validation_negative_eps() {
    let err = VitConfig::new(3, 768, 12, 12, 3072, 16, 224, -1e-6, true);
    assert!(err.is_err(), "negative layer_norm_eps must be rejected");
}

// ---------------------------------------------------------------------------
// VitConfig seq_len calculation
// ---------------------------------------------------------------------------

#[test]
fn test_vit_config_seq_len_calculation() {
    // ViT-B/16 with 224x224: num_patches = (224/16)^2 = 14^2 = 196
    // With CLS: seq_len = 196 + 1 = 197
    let cfg = VitConfig::new(3, 768, 12, 12, 3072, 16, 224, 1e-6, true).unwrap();
    assert_eq!(
        cfg.seq_len(),
        197,
        "ViT-B/16 with CLS: (224/16)^2 + 1 = 197"
    );

    // Without CLS: seq_len = 196
    let cfg_no_cls = VitConfig::new(3, 768, 12, 12, 3072, 16, 224, 1e-6, false).unwrap();
    assert_eq!(
        cfg_no_cls.seq_len(),
        196,
        "ViT-B/16 without CLS: (224/16)^2 = 196"
    );

    // ViT-L/32 with 384x384: num_patches = (384/32)^2 = 12^2 = 144
    let cfg_l32 = VitConfig::new(3, 1024, 24, 16, 4096, 32, 384, 1e-6, true).unwrap();
    assert_eq!(
        cfg_l32.seq_len(),
        145,
        "ViT-L/32 with CLS: (384/32)^2 + 1 = 145"
    );
}

// ---------------------------------------------------------------------------
// VitConfig num_patches
// ---------------------------------------------------------------------------

#[test]
fn test_vit_config_num_patches() {
    // (224/16)^2 = 196
    let cfg = VitConfig::new(3, 768, 12, 12, 3072, 16, 224, 1e-6, false).unwrap();
    assert_eq!(cfg.num_patches(), 196);

    // (384/16)^2 = 576
    let cfg2 = VitConfig::new(3, 768, 12, 12, 3072, 16, 384, 1e-6, false).unwrap();
    assert_eq!(cfg2.num_patches(), 576);

    // (256/32)^2 = 64
    let cfg3 = VitConfig::new(3, 1024, 24, 16, 4096, 32, 256, 1e-6, false).unwrap();
    assert_eq!(cfg3.num_patches(), 64);

    // Minimal: (1/1)^2 = 1
    let cfg4 = VitConfig::new(1, 64, 1, 1, 256, 1, 1, 1e-6, false).unwrap();
    assert_eq!(cfg4.num_patches(), 1);
}

// ---------------------------------------------------------------------------
// SigLip2Config validation
// ---------------------------------------------------------------------------

#[test]
fn test_siglip2_config_validation_valid() {
    let cfg = SigLip2Config::new(3, 768, 12, 12, 3072, 16, 224, 1e-6);
    assert!(cfg.is_ok(), "valid SigLip2 config should succeed");
}

#[test]
fn test_siglip2_config_validation_zero_patch() {
    let err = SigLip2Config::new(3, 768, 12, 12, 3072, 0, 224, 1e-6);
    assert!(err.is_err(), "SigLip2 patch_size=0 must fail");
}

#[test]
fn test_siglip2_config_validation_indivisible() {
    let err = SigLip2Config::new(3, 768, 12, 12, 3072, 16, 225, 1e-6);
    assert!(
        err.is_err(),
        "SigLip2 image_size not divisible by patch_size must fail"
    );
}

#[test]
fn test_siglip2_config_validation_zero_hidden() {
    let err = SigLip2Config::new(3, 0, 12, 12, 3072, 16, 224, 1e-6);
    assert!(err.is_err(), "SigLip2 hidden_size=0 must fail");
}

#[test]
fn test_siglip2_config_validation_indivisible_heads() {
    // 768 not divisible by 7
    let err = SigLip2Config::new(3, 768, 12, 7, 3072, 16, 224, 1e-6);
    assert!(
        err.is_err(),
        "SigLip2 hidden not divisible by heads must fail"
    );
}

// ---------------------------------------------------------------------------
// SigLip2Config -> VitConfig conversion
// ---------------------------------------------------------------------------

#[test]
fn test_siglip2_to_vit_config() {
    let siglip = SigLip2Config::new(3, 768, 12, 12, 3072, 16, 224, 1e-6).unwrap();
    let vit = siglip.to_vit_config().unwrap();

    assert_eq!(vit.num_channels, 3);
    assert_eq!(vit.hidden_size, 768);
    assert_eq!(vit.num_layers, 12);
    assert_eq!(vit.num_heads, 12);
    assert_eq!(vit.intermediate_size, 3072);
    assert_eq!(vit.patch_size, 16);
    assert_eq!(vit.image_size, 224);
    assert!((vit.layer_norm_eps - 1e-6).abs() < 1e-12);
    // SigLIP2 always sets use_cls_token = false
    assert!(!vit.use_cls_token, "SigLip2 must set use_cls_token=false");
}

// ---------------------------------------------------------------------------
// SigLip2Config base_patch16 factory
// ---------------------------------------------------------------------------

#[test]
fn test_siglip2_base_patch16() {
    let cfg = SigLip2Config::base_patch16(224).unwrap();
    assert_eq!(cfg.num_channels, 3);
    assert_eq!(cfg.hidden_size, 768);
    assert_eq!(cfg.num_layers, 12);
    assert_eq!(cfg.num_heads, 12);
    assert_eq!(cfg.intermediate_size, 3072);
    assert_eq!(cfg.patch_size, 16);
    assert_eq!(cfg.image_size, 224);
    assert!((cfg.layer_norm_eps - 1e-6).abs() < 1e-12);

    // Non-standard image size should also work
    let cfg2 = SigLip2Config::base_patch16(384).unwrap();
    assert_eq!(cfg2.image_size, 384);

    // Image size not divisible by 16 should fail
    let err = SigLip2Config::base_patch16(225);
    assert!(
        err.is_err(),
        "base_patch16(225) should fail: 225 not divisible by 16"
    );
}

// ---------------------------------------------------------------------------
// VitConfig head_dim
// ---------------------------------------------------------------------------

#[test]
fn test_vit_config_head_dim() {
    // ViT-B/16: hidden=768, heads=12 -> head_dim=64
    let cfg = VitConfig::new(3, 768, 12, 12, 3072, 16, 224, 1e-6, true).unwrap();
    assert_eq!(cfg.hidden_size / cfg.num_heads, 64);

    // ViT-L/16: hidden=1024, heads=16 -> head_dim=64
    let cfg2 = VitConfig::new(3, 1024, 24, 16, 4096, 16, 224, 1e-6, true).unwrap();
    assert_eq!(cfg2.hidden_size / cfg2.num_heads, 64);

    // ViT-H: hidden=1280, heads=16 -> head_dim=80
    let cfg3 = VitConfig::new(3, 1280, 32, 16, 5120, 16, 224, 1e-6, true).unwrap();
    assert_eq!(cfg3.hidden_size / cfg3.num_heads, 80);

    // Qwen2VL has an explicit head_dim() method
    let q_cfg = Qwen2VLVitConfig::new(3, 1280, 32, 16, 5120, 14, 2, 1e-6, 14, vec![]).unwrap();
    assert_eq!(q_cfg.head_dim(), 80, "1280 / 16 = 80");

    // Qwen3VL also has head_dim()
    let q3_cfg = Qwen3VLVitConfig::new(
        3,
        3584,
        32,
        28,
        18944,
        14,
        2,
        1e-6,
        14,
        4,
        vec![7, 15, 23, 31],
        3584,
    )
    .unwrap();
    assert_eq!(q3_cfg.head_dim(), 128, "3584 / 28 = 128");
}

// ---------------------------------------------------------------------------
// PoolingStrategy Debug formatting
// ---------------------------------------------------------------------------

#[test]
fn test_pooling_strategy_debug() {
    assert_eq!(format!("{:?}", PoolingStrategy::Cls), "Cls");
    assert_eq!(format!("{:?}", PoolingStrategy::Mean), "Mean");
    assert_eq!(format!("{:?}", PoolingStrategy::None), "None");
}
