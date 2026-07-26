// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::dyn_tensor::DynTensor;
use crate::layers::vision::detect_head::make_anchor_grid;
use crate::layers::vision::{ConvBnAct, DetectHead, ScaleOutput};
use crate::layers::{Activation, BatchNorm, Conv2d, Conv2dConfig};
use crate::{DType, Device};

fn make_conv_bn(in_c: usize, out_c: usize, k: usize) -> ConvBnAct {
    let padding = k / 2;
    let weight = DynTensor::full(&[out_c, in_c, k, k], 0.01, DType::F32, &Device::Cpu).unwrap();
    let cfg = Conv2dConfig::new(padding, 1, 1);
    let conv = Conv2d::new(weight, None, cfg).unwrap();
    let bn = BatchNorm::new(
        DynTensor::zeros(&[out_c], DType::F32, &Device::Cpu).unwrap(),
        DynTensor::ones(&[out_c], DType::F32, &Device::Cpu).unwrap(),
        Some(DynTensor::ones(&[out_c], DType::F32, &Device::Cpu).unwrap()),
        Some(DynTensor::zeros(&[out_c], DType::F32, &Device::Cpu).unwrap()),
        1e-5,
    )
    .unwrap();
    ConvBnAct::new(conv, bn, Some(Activation::Silu))
}

fn make_conv2d(in_c: usize, out_c: usize) -> Conv2d {
    let weight = DynTensor::full(&[out_c, in_c, 1, 1], 0.01, DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[out_c], DType::F32, &Device::Cpu).unwrap();
    let cfg = Conv2dConfig::new(0, 1, 1);
    Conv2d::new(weight, Some(bias), cfg).unwrap()
}

fn make_detect_head(
    in_channels: &[usize],
    num_classes: usize,
    reg_max: usize,
    hidden: usize,
) -> DetectHead {
    let mut cls_convs = Vec::new();
    let mut reg_convs = Vec::new();
    let mut cls_preds = Vec::new();
    let mut reg_preds = Vec::new();

    for &in_c in in_channels {
        cls_convs.push([
            make_conv_bn(in_c, hidden, 3),
            make_conv_bn(hidden, hidden, 3),
        ]);
        reg_convs.push([
            make_conv_bn(in_c, hidden, 3),
            make_conv_bn(hidden, hidden, 3),
        ]);
        cls_preds.push(make_conv2d(hidden, num_classes));
        reg_preds.push(make_conv2d(hidden, 4 * reg_max));
    }

    DetectHead::new(
        cls_convs,
        reg_convs,
        cls_preds,
        reg_preds,
        num_classes,
        reg_max,
    )
    .unwrap()
}

#[test]
fn test_detect_head_output_shapes() {
    let num_classes = 80;
    let reg_max = 16;
    let hidden = 16;
    let in_channels = [16, 32, 64];
    let head = make_detect_head(&in_channels, num_classes, reg_max, hidden);

    let f3 = DynTensor::full(&[1, 16, 8, 8], 0.5, DType::F32, &Device::Cpu).unwrap();
    let f4 = DynTensor::full(&[1, 32, 4, 4], 0.5, DType::F32, &Device::Cpu).unwrap();
    let f5 = DynTensor::full(&[1, 64, 2, 2], 0.5, DType::F32, &Device::Cpu).unwrap();

    let outputs = head.forward_multi(&[&f3, &f4, &f5]).unwrap();
    assert_eq!(outputs.len(), 3);

    assert_eq!(outputs[0].cls_logits.dims(), &[1, 80, 8, 8]);
    assert_eq!(outputs[0].reg_preds.dims(), &[1, 64, 8, 8]); // 4 * 16

    assert_eq!(outputs[1].cls_logits.dims(), &[1, 80, 4, 4]);
    assert_eq!(outputs[1].reg_preds.dims(), &[1, 64, 4, 4]);

    assert_eq!(outputs[2].cls_logits.dims(), &[1, 80, 2, 2]);
    assert_eq!(outputs[2].reg_preds.dims(), &[1, 64, 2, 2]);
}

#[test]
fn test_detect_head_wrong_scale_count() {
    let head = make_detect_head(&[64, 128], 10, 16, 32);
    let f1 = DynTensor::full(&[1, 64, 8, 8], 0.5, DType::F32, &Device::Cpu).unwrap();
    // Only 1 feature but head expects 2 — should error
    let result = head.forward_multi(&[&f1]);
    assert!(result.is_err());
}

#[test]
fn test_detect_head_batch() {
    let head = make_detect_head(&[32], 5, 8, 32);
    let feat = DynTensor::full(&[4, 32, 10, 10], 0.5, DType::F32, &Device::Cpu).unwrap();
    let outputs = head.forward_multi(&[&feat]).unwrap();
    assert_eq!(outputs[0].cls_logits.dims(), &[4, 5, 10, 10]);
    assert_eq!(outputs[0].reg_preds.dims(), &[4, 32, 10, 10]); // 4 * 8
}

#[test]
fn test_decode_dfl_shape() {
    let head = make_detect_head(&[32], 5, 16, 32);
    // Simulate regression output: [B, 4*16, H, W]
    let reg = DynTensor::full(&[1, 64, 10, 10], 0.1, DType::F32, &Device::Cpu).unwrap();
    let decoded = head.decode_dfl(&reg).unwrap();
    assert_eq!(decoded.dims(), &[1, 4, 10, 10]);
}

#[test]
fn test_decode_dfl_values_in_range() {
    let head = make_detect_head(&[32], 5, 16, 32);
    let reg = DynTensor::full(&[1, 64, 4, 4], 0.0, DType::F32, &Device::Cpu).unwrap();
    let decoded = head.decode_dfl(&reg).unwrap();
    // With uniform input, softmax gives 1/reg_max per bin.
    // Integral = sum(i * 1/16) for i in 0..16 = (15*16/2)/16 = 7.5
    let vals = decoded.to_flat_vec::<f32>().unwrap();
    for &v in &vals {
        assert!((v - 7.5).abs() < 0.01, "expected ~7.5, got {v}");
    }
}

// ============================================================================
// Anchor grid tests
// ============================================================================

#[test]
fn test_make_anchor_grid_shape() {
    let (gx, gy) = make_anchor_grid(4, 6, &Device::Cpu).unwrap();
    assert_eq!(gx.dims(), &[1, 1, 4, 6]);
    assert_eq!(gy.dims(), &[1, 1, 4, 6]);
}

#[test]
fn test_make_anchor_grid_values() {
    let (gx, gy) = make_anchor_grid(2, 3, &Device::Cpu).unwrap();
    let gx_v = gx.to_flat_vec::<f32>().unwrap();
    let gy_v = gy.to_flat_vec::<f32>().unwrap();
    // Row 0: (col=0,1,2), Row 1: (col=0,1,2)
    assert_eq!(gx_v, vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
    assert_eq!(gy_v, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_make_anchor_grid_zero_dims_errors() {
    assert!(make_anchor_grid(0, 4, &Device::Cpu).is_err());
    assert!(make_anchor_grid(4, 0, &Device::Cpu).is_err());
}

// ============================================================================
// DocLayout-YOLO specific tests: 10 classes, 3 scales, reg_max=16
// ============================================================================

/// DocLayout-YOLO document element class labels.
const DOCLAYOUT_CLASSES: [&str; 10] = [
    "title",
    "text",
    "list",
    "table",
    "figure",
    "caption",
    "header",
    "footer",
    "reference",
    "equation",
];

const DOCLAYOUT_NUM_CLASSES: usize = 10;
const DOCLAYOUT_REG_MAX: usize = 16;
const DOCLAYOUT_STRIDES: [usize; 3] = [8, 16, 32];
const DOCLAYOUT_HIDDEN: usize = 64;

fn make_doclayout_head(in_channels: &[usize]) -> DetectHead {
    make_detect_head(
        in_channels,
        DOCLAYOUT_NUM_CLASSES,
        DOCLAYOUT_REG_MAX,
        DOCLAYOUT_HIDDEN,
    )
}

#[test]
fn test_doclayout_yolo_forward_shapes() {
    // DocLayout-YOLO with 3 scales: P3 (stride 8), P4 (stride 16), P5 (stride 32)
    let in_channels = [64, 128, 256];
    let head = make_doclayout_head(&in_channels);

    assert_eq!(head.num_classes(), DOCLAYOUT_NUM_CLASSES);
    assert_eq!(head.reg_max(), DOCLAYOUT_REG_MAX);
    assert_eq!(head.num_scales(), 3);

    // Simulate 640x640 input: P3=80x80, P4=40x40, P5=20x20
    let p3 = DynTensor::full(&[1, 64, 80, 80], 0.1, DType::F32, &Device::Cpu).unwrap();
    let p4 = DynTensor::full(&[1, 128, 40, 40], 0.1, DType::F32, &Device::Cpu).unwrap();
    let p5 = DynTensor::full(&[1, 256, 20, 20], 0.1, DType::F32, &Device::Cpu).unwrap();

    let outputs = head.forward_multi(&[&p3, &p4, &p5]).unwrap();
    assert_eq!(outputs.len(), 3);

    // P3 scale: [1, 10, 80, 80] cls + [1, 64, 80, 80] reg
    assert_eq!(outputs[0].cls_logits.dims(), &[1, 10, 80, 80]);
    assert_eq!(outputs[0].reg_preds.dims(), &[1, 64, 80, 80]);

    // P4 scale: [1, 10, 40, 40] cls + [1, 64, 40, 40] reg
    assert_eq!(outputs[1].cls_logits.dims(), &[1, 10, 40, 40]);
    assert_eq!(outputs[1].reg_preds.dims(), &[1, 64, 40, 40]);

    // P5 scale: [1, 10, 20, 20] cls + [1, 64, 20, 20] reg
    assert_eq!(outputs[2].cls_logits.dims(), &[1, 10, 20, 20]);
    assert_eq!(outputs[2].reg_preds.dims(), &[1, 64, 20, 20]);
}

#[test]
fn test_doclayout_yolo_total_anchor_count() {
    // 640x640 image: 80*80 + 40*40 + 20*20 = 6400 + 1600 + 400 = 8400 anchors
    let total = 80 * 80 + 40 * 40 + 20 * 20;
    assert_eq!(
        total, 8400,
        "DocLayout-YOLO 640px should have 8400 anchor positions"
    );
}

#[test]
fn test_doclayout_yolo_dfl_decode_per_scale() {
    let head = make_doclayout_head(&[64, 128, 256]);

    // P3 scale regression: [1, 64, 80, 80] -> [1, 4, 80, 80]
    let reg_p3 = DynTensor::full(&[1, 64, 80, 80], 0.0, DType::F32, &Device::Cpu).unwrap();
    let decoded = head.decode_dfl(&reg_p3).unwrap();
    assert_eq!(decoded.dims(), &[1, 4, 80, 80]);

    // P5 scale regression: [1, 64, 20, 20] -> [1, 4, 20, 20]
    let reg_p5 = DynTensor::full(&[1, 64, 20, 20], 0.0, DType::F32, &Device::Cpu).unwrap();
    let decoded = head.decode_dfl(&reg_p5).unwrap();
    assert_eq!(decoded.dims(), &[1, 4, 20, 20]);
}

#[test]
fn test_doclayout_yolo_decode_detections_basic() {
    // Small scale test: 3 scales at 4x4, 2x2, 1x1
    let in_channels = [32, 64, 128];
    let head = make_doclayout_head(&in_channels);

    // Simulate feature maps and run forward
    let p3 = DynTensor::full(&[1, 32, 4, 4], 0.1, DType::F32, &Device::Cpu).unwrap();
    let p4 = DynTensor::full(&[1, 64, 2, 2], 0.1, DType::F32, &Device::Cpu).unwrap();
    let p5 = DynTensor::full(&[1, 128, 1, 1], 0.1, DType::F32, &Device::Cpu).unwrap();

    let scale_outputs = head.forward_multi(&[&p3, &p4, &p5]).unwrap();
    let img_size = (32, 32);

    // With very low confidence threshold, should get some detections
    let detections = head
        .decode_detections(&scale_outputs, &DOCLAYOUT_STRIDES, img_size, 0.01, 0.45)
        .unwrap();

    // Verify all detection coordinates are within image bounds
    for det in &detections {
        assert!(
            det.x1 >= 0.0 && det.x1 <= img_size.1 as f32,
            "x1 out of bounds: {}",
            det.x1
        );
        assert!(
            det.y1 >= 0.0 && det.y1 <= img_size.0 as f32,
            "y1 out of bounds: {}",
            det.y1
        );
        assert!(
            det.x2 >= 0.0 && det.x2 <= img_size.1 as f32,
            "x2 out of bounds: {}",
            det.x2
        );
        assert!(
            det.y2 >= 0.0 && det.y2 <= img_size.0 as f32,
            "y2 out of bounds: {}",
            det.y2
        );
        assert!(
            det.confidence >= 0.01,
            "confidence below threshold: {}",
            det.confidence
        );
        assert!(
            (det.class_id as usize) < DOCLAYOUT_NUM_CLASSES,
            "class_id out of range: {}",
            det.class_id
        );
    }
}

#[test]
fn test_doclayout_yolo_decode_detections_high_threshold_filters() {
    let in_channels = [32, 64, 128];
    let head = make_doclayout_head(&in_channels);

    let p3 = DynTensor::full(&[1, 32, 4, 4], 0.1, DType::F32, &Device::Cpu).unwrap();
    let p4 = DynTensor::full(&[1, 64, 2, 2], 0.1, DType::F32, &Device::Cpu).unwrap();
    let p5 = DynTensor::full(&[1, 128, 1, 1], 0.1, DType::F32, &Device::Cpu).unwrap();

    let scale_outputs = head.forward_multi(&[&p3, &p4, &p5]).unwrap();

    // With very high confidence threshold, small random-like weights should
    // produce zero detections (sigmoid of small values < 0.99)
    let detections = head
        .decode_detections(&scale_outputs, &DOCLAYOUT_STRIDES, (32, 32), 0.99, 0.45)
        .unwrap();

    assert!(
        detections.is_empty(),
        "expected no detections with 0.99 threshold, got {}",
        detections.len()
    );
}

#[test]
fn test_doclayout_yolo_decode_detections_stride_mismatch_errors() {
    let head = make_doclayout_head(&[32, 64, 128]);
    let p3 = DynTensor::full(&[1, 32, 4, 4], 0.1, DType::F32, &Device::Cpu).unwrap();
    let p4 = DynTensor::full(&[1, 64, 2, 2], 0.1, DType::F32, &Device::Cpu).unwrap();
    let p5 = DynTensor::full(&[1, 128, 1, 1], 0.1, DType::F32, &Device::Cpu).unwrap();

    let scale_outputs = head.forward_multi(&[&p3, &p4, &p5]).unwrap();

    // Pass only 2 strides for 3 scale outputs
    let result = head.decode_detections(&scale_outputs, &[8, 16], (32, 32), 0.25, 0.45);
    assert!(result.is_err(), "stride/scale count mismatch should error");
}

#[test]
fn test_doclayout_yolo_decode_with_synthetic_high_confidence() {
    // Construct ScaleOutput directly with known values to verify
    // the decoding pipeline produces correct bounding boxes.
    let in_channels = [32];
    let head = make_doclayout_head(&in_channels);

    // Create a 1x1 spatial feature: one anchor point at grid (0, 0).
    // Classification: make class 3 ("table") have high logit (+5.0),
    // all others at -5.0. sigmoid(5.0) ~= 0.993.
    let mut cls_data = vec![-5.0f32; DOCLAYOUT_NUM_CLASSES];
    cls_data[3] = 5.0; // table class
    let cls_logits =
        DynTensor::from_vec(cls_data, &[1, DOCLAYOUT_NUM_CLASSES, 1, 1], &Device::Cpu).unwrap();

    // Regression: uniform 0.0 across all 64 channels -> DFL decodes to 7.5
    // for each of (left, top, right, bottom) distances.
    let reg_preds = DynTensor::full(
        &[1, 4 * DOCLAYOUT_REG_MAX, 1, 1],
        0.0,
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();

    let scale_output = ScaleOutput {
        cls_logits,
        reg_preds,
    };

    // Decode with stride=8, image 640x640
    let detections = head
        .decode_detections(&[scale_output], &[8], (640, 640), 0.5, 0.45)
        .unwrap();

    assert_eq!(detections.len(), 1, "should produce exactly one detection");
    let det = &detections[0];
    assert_eq!(det.class_id, 3, "should detect class 3 (table)");
    assert!(
        (det.confidence - 0.993).abs() < 0.01,
        "confidence should be ~sigmoid(5.0)=0.993, got {}",
        det.confidence
    );

    // Grid (0,0) with stride 8: center = (0.5 * 8, 0.5 * 8) = (4.0, 4.0)
    // DFL decode of uniform 0.0 = 7.5 for all four distances
    // x1 = (0 + 0.5 - 7.5) * 8 = -56.0, clamped to 0.0
    // y1 = (0 + 0.5 - 7.5) * 8 = -56.0, clamped to 0.0
    // x2 = (0 + 0.5 + 7.5) * 8 = 64.0
    // y2 = (0 + 0.5 + 7.5) * 8 = 64.0
    assert!(
        (det.x1 - 0.0).abs() < 0.01,
        "x1 should be clamped to 0.0, got {}",
        det.x1
    );
    assert!(
        (det.y1 - 0.0).abs() < 0.01,
        "y1 should be clamped to 0.0, got {}",
        det.y1
    );
    assert!(
        (det.x2 - 64.0).abs() < 0.01,
        "x2 should be 64.0, got {}",
        det.x2
    );
    assert!(
        (det.y2 - 64.0).abs() < 0.01,
        "y2 should be 64.0, got {}",
        det.y2
    );
}

#[test]
fn test_doclayout_class_count_matches() {
    assert_eq!(DOCLAYOUT_CLASSES.len(), DOCLAYOUT_NUM_CLASSES);
}

#[test]
fn test_doclayout_yolo_nms_suppresses_overlapping() {
    // Test that NMS properly suppresses overlapping detections from
    // multiple scales in the DocLayout-YOLO pipeline.
    let in_channels = [32, 64];
    let head = make_doclayout_head(&in_channels);

    // Two 1x1 scales: both produce overlapping boxes for the same class
    // Scale 0 (stride 8): grid (0,0) -> center (4, 4)
    let mut cls0 = vec![-5.0f32; DOCLAYOUT_NUM_CLASSES];
    cls0[0] = 4.0; // title, sigmoid(4.0) ~= 0.982
    let cls0_t =
        DynTensor::from_vec(cls0, &[1, DOCLAYOUT_NUM_CLASSES, 1, 1], &Device::Cpu).unwrap();
    let reg0_t = DynTensor::full(
        &[1, 4 * DOCLAYOUT_REG_MAX, 1, 1],
        0.0,
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();

    // Scale 1 (stride 16): grid (0,0) -> center (8, 8)
    let mut cls1 = vec![-5.0f32; DOCLAYOUT_NUM_CLASSES];
    cls1[0] = 3.0; // title, sigmoid(3.0) ~= 0.953
    let cls1_t =
        DynTensor::from_vec(cls1, &[1, DOCLAYOUT_NUM_CLASSES, 1, 1], &Device::Cpu).unwrap();
    let reg1_t = DynTensor::full(
        &[1, 4 * DOCLAYOUT_REG_MAX, 1, 1],
        0.0,
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();

    let scale_outputs = vec![
        ScaleOutput {
            cls_logits: cls0_t,
            reg_preds: reg0_t,
        },
        ScaleOutput {
            cls_logits: cls1_t,
            reg_preds: reg1_t,
        },
    ];

    let detections = head
        .decode_detections(&scale_outputs, &[8, 16], (640, 640), 0.5, 0.45)
        .unwrap();

    // Both boxes overlap significantly. NMS should suppress the lower-confidence one.
    assert!(
        detections.len() <= 2,
        "NMS should limit detections, got {}",
        detections.len()
    );

    // All surviving detections should be class 0 (title)
    for det in &detections {
        assert_eq!(det.class_id, 0, "surviving detection should be title class");
    }

    // Highest confidence detection should be first
    if !detections.is_empty() {
        assert!(
            detections[0].confidence > 0.95,
            "first detection should have highest confidence"
        );
    }
}
