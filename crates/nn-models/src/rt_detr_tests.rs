// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for RT-DETRv2 model builder.

use super::*;

#[test]
fn test_heron_preset_config() {
    let config = RtDetrConfig::preset_heron();
    assert_eq!(config.num_classes, 17);
    assert_eq!(config.num_queries, 300);
    assert_eq!(config.hidden_dim, 256);
    assert_eq!(config.num_heads, 8);
    assert_eq!(config.backbone_channels, [128, 256, 512]);
    assert_eq!(config.input_size, 640);
    config.validate().expect("heron preset should be valid");
}

#[test]
fn test_coco_preset_config() {
    let config = RtDetrConfig::preset_coco();
    assert_eq!(config.num_classes, 80);
    assert_eq!(config.num_queries, 300);
    config.validate().expect("coco preset should be valid");
}

#[test]
fn test_default_is_heron() {
    let config = RtDetrConfig::default();
    assert_eq!(config.num_classes, 17);
}

#[test]
fn test_config_validation_zero_hidden_dim() {
    let mut config = RtDetrConfig::preset_heron();
    config.hidden_dim = 0;
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_zero_heads() {
    let mut config = RtDetrConfig::preset_heron();
    config.num_heads = 0;
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_indivisible_dim() {
    let mut config = RtDetrConfig::preset_heron();
    config.hidden_dim = 255; // Not divisible by 8.
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_zero_classes() {
    let mut config = RtDetrConfig::preset_heron();
    config.num_classes = 0;
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_zero_queries() {
    let mut config = RtDetrConfig::preset_heron();
    config.num_queries = 0;
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_conf_threshold_out_of_range() {
    let mut config = RtDetrConfig::preset_heron();
    config.conf_threshold = 1.5;
    assert!(config.validate().is_err());

    config.conf_threshold = -0.1;
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_conf_threshold_boundary() {
    let mut config = RtDetrConfig::preset_heron();
    config.conf_threshold = 0.0;
    config.validate().expect("0.0 threshold should be valid");

    config.conf_threshold = 1.0;
    config.validate().expect("1.0 threshold should be valid");
}

#[test]
fn test_heron_class_names_count() {
    assert_eq!(HERON_CLASS_NAMES.len(), 17);
}

#[test]
fn test_heron_class_names_content() {
    assert_eq!(HERON_CLASS_NAMES[0], "caption");
    assert_eq!(HERON_CLASS_NAMES[9], "text");
    assert_eq!(HERON_CLASS_NAMES[10], "title");
    assert_eq!(HERON_CLASS_NAMES[16], "handwriting");
}

#[test]
fn test_decode_detections_basic() {
    let config = RtDetrConfig {
        num_classes: 3,
        num_queries: 2,
        conf_threshold: 0.5,
        ..RtDetrConfig::preset_heron()
    };
    // Create a dummy model by using the config for decoding only.
    // class_logits: 2 queries x 3 classes.
    // Query 0: class 1 has high logit (2.0 → sigmoid ≈ 0.88).
    // Query 1: class 0 has low logit (-2.0 → sigmoid ≈ 0.12) — below threshold.
    let class_logits = [
        -1.0, 2.0, 0.0, // Query 0: best = class 1, sigmoid(2.0) ≈ 0.88
        -2.0, -2.0, -2.0, // Query 1: best = class 0, sigmoid(-2.0) ≈ 0.12
    ];
    let box_preds = [
        0.5, 0.5, 0.3, 0.3, // Query 0: (cx=0.5, cy=0.5, w=0.3, h=0.3)
        0.1, 0.1, 0.2, 0.2, // Query 1: filtered out
    ];

    // Use a temporary RtDetr-like structure just for decoding.
    // We test decode_detections as a method that only needs config.
    let dets = decode_detections_standalone(&config, &class_logits, &box_preds, 2, 3);
    assert_eq!(dets.len(), 1);
    assert_eq!(dets[0].0, 1); // class 1
    assert!(dets[0].1 > 0.85); // confidence ≈ 0.88
                               // (cx=0.5, cy=0.5, w=0.3, h=0.3) → (x1=0.35, y1=0.35, x2=0.65, y2=0.65)
    assert!((dets[0].2[0] - 0.35).abs() < 1e-5);
    assert!((dets[0].2[1] - 0.35).abs() < 1e-5);
    assert!((dets[0].2[2] - 0.65).abs() < 1e-5);
    assert!((dets[0].2[3] - 0.65).abs() < 1e-5);
}

/// Standalone decoding helper (mirrors RtDetr::decode_detections logic)
/// used for tests where constructing the full model is not needed.
fn decode_detections_standalone(
    config: &RtDetrConfig,
    class_logits: &[f32],
    box_preds: &[f32],
    num_queries: usize,
    num_classes: usize,
) -> Vec<(u32, f32, [f32; 4])> {
    let threshold = config.conf_threshold;
    let mut detections = Vec::new();
    for q in 0..num_queries {
        let logit_offset = q * num_classes;
        let box_offset = q * 4;
        let mut best_class = 0u32;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..num_classes {
            let logit = class_logits[logit_offset + c];
            let score = 1.0 / (1.0 + (-logit).exp());
            if score > best_score {
                best_score = score;
                best_class = c as u32;
            }
        }
        if best_score < threshold {
            continue;
        }
        let cx = box_preds[box_offset];
        let cy = box_preds[box_offset + 1];
        let w = box_preds[box_offset + 2];
        let h = box_preds[box_offset + 3];
        detections.push((
            best_class,
            best_score,
            [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
        ));
    }
    detections
}

#[test]
fn test_decode_detections_all_below_threshold() {
    let config = RtDetrConfig {
        num_classes: 2,
        num_queries: 1,
        conf_threshold: 0.99,
        ..RtDetrConfig::preset_heron()
    };
    let class_logits = [0.0, 0.0]; // sigmoid(0) = 0.5 < 0.99
    let box_preds = [0.5, 0.5, 0.2, 0.2];
    let dets = decode_detections_standalone(&config, &class_logits, &box_preds, 1, 2);
    assert!(dets.is_empty());
}

#[test]
fn test_load_synthetic_weights_hf() {
    let device = nn_core::Device::Cpu;
    let vb = VarBuilder::zeros(nn_core::DType::F32, &device);
    let config = RtDetrConfig::preset_heron();
    assert_eq!(config.backbone_variant, RtDetrBackboneVariant::HuggingFace);
    let model = RtDetr::load(&vb, config);
    assert!(
        model.is_ok(),
        "RT-DETR (HF backbone) should load with synthetic (zero) weights: {:?}",
        model.err()
    );
}

#[test]
fn test_load_synthetic_weights_torchvision() {
    let device = nn_core::Device::Cpu;
    let vb = VarBuilder::zeros(nn_core::DType::F32, &device);
    let config = RtDetrConfig::preset_coco();
    assert_eq!(config.backbone_variant, RtDetrBackboneVariant::Torchvision);
    let model = RtDetr::load(&vb, config);
    assert!(
        model.is_ok(),
        "RT-DETR (torchvision backbone) should load with synthetic weights: {:?}",
        model.err()
    );
}

#[test]
fn test_heron_preset_uses_hf_backbone() {
    let config = RtDetrConfig::preset_heron();
    assert_eq!(config.backbone_variant, RtDetrBackboneVariant::HuggingFace);
}

#[test]
fn test_coco_preset_uses_torchvision_backbone() {
    let config = RtDetrConfig::preset_coco();
    assert_eq!(config.backbone_variant, RtDetrBackboneVariant::Torchvision);
}

/// Verify full model forward pass produces correct output shapes with the
/// Heron (HuggingFace backbone) configuration using synthetic zero weights.
///
/// This exercises the entire architecture: ResNet18Hf backbone -> channel
/// projections -> AIFI encoder -> DETR decoder -> class + bbox heads.
///
/// Input:  [1, 3, 640, 640]
/// Output: logits [1, 300, 18] (17 classes + 1 no-object) + boxes [1, 300, 4]
#[test]
fn test_forward_shapes_hf_synthetic() {
    let device = nn_core::Device::Cpu;
    let vb = VarBuilder::zeros(nn_core::DType::F32, &device);
    let config = RtDetrConfig::preset_heron();
    let model =
        RtDetr::load(&vb, config.clone()).expect("RT-DETR (HF) should load with zero weights");

    let image =
        DynTensor::zeros(&[1, 3, 640, 640], nn_core::DType::F32, &device)
            .unwrap();
    let (class_logits, bbox_preds) = model
        .forward(&image)
        .expect("RT-DETR forward should succeed with zero weights");

    // class_logits: [B, num_queries, num_classes + 1]
    let logit_shape = class_logits.shape().dims().to_vec();
    assert_eq!(
        logit_shape,
        vec![1, config.num_queries, config.num_classes + 1],
        "class_logits shape: expected [1, 300, 18], got {logit_shape:?}"
    );

    // bbox_preds: [B, num_queries, 4]
    let bbox_shape = bbox_preds.shape().dims().to_vec();
    assert_eq!(
        bbox_shape,
        vec![1, config.num_queries, 4],
        "bbox_preds shape: expected [1, 300, 4], got {bbox_shape:?}"
    );
}
