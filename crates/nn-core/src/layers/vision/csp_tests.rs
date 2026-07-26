// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::dyn_tensor::DynTensor;
use crate::layers::vision::{Bottleneck, C2f, ConvBnAct};
use crate::layers::{Activation, BatchNorm, Conv2d, Conv2dConfig, Module};
use crate::{DType, Device};

/// Helper: create a ConvBnAct from channel counts without VarBuilder.
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

#[test]
fn test_bottleneck_forward_shape() {
    let ch = 32;
    let cv1 = make_conv_bn(ch, ch, 3);
    let cv2 = make_conv_bn(ch, ch, 3);
    let bottleneck = Bottleneck::new(cv1, cv2, true);

    let input = DynTensor::full(&[1, 32, 16, 16], 0.5, DType::F32, &Device::Cpu).unwrap();
    let output = bottleneck.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 32, 16, 16]);
}

#[test]
fn test_bottleneck_shortcut_vs_no_shortcut() {
    let ch = 16;
    let input = DynTensor::full(&[1, 16, 8, 8], 0.5, DType::F32, &Device::Cpu).unwrap();

    // With shortcut
    let cv1 = make_conv_bn(ch, ch, 3);
    let cv2 = make_conv_bn(ch, ch, 3);
    let with_sc = Bottleneck::new(cv1, cv2, true);
    let out_sc = with_sc.forward(&input).unwrap();
    assert_eq!(out_sc.dims(), &[1, 16, 8, 8]);

    // Without shortcut
    let cv1 = make_conv_bn(ch, ch, 3);
    let cv2 = make_conv_bn(ch, ch, 3);
    let no_sc = Bottleneck::new(cv1, cv2, false);
    let out_no_sc = no_sc.forward(&input).unwrap();
    assert_eq!(out_no_sc.dims(), &[1, 16, 8, 8]);
}

#[test]
fn test_c2f_forward_shape() {
    let in_c = 64;
    let out_c = 64;
    let n_bottlenecks = 2;
    let hidden = out_c / 2;

    let cv1 = make_conv_bn(in_c, 2 * hidden, 1);
    let cat_channels = (2 + n_bottlenecks) * hidden;
    let cv2 = make_conv_bn(cat_channels, out_c, 1);

    let bottlenecks = (0..n_bottlenecks)
        .map(|_| {
            let b_cv1 = make_conv_bn(hidden, hidden, 3);
            let b_cv2 = make_conv_bn(hidden, hidden, 3);
            Bottleneck::new(b_cv1, b_cv2, true)
        })
        .collect();

    let c2f = C2f::new(cv1, cv2, bottlenecks);
    let input = DynTensor::full(&[1, 64, 20, 20], 0.5, DType::F32, &Device::Cpu).unwrap();
    let output = c2f.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 64, 20, 20]);
}

#[test]
fn test_c2f_different_in_out_channels() {
    let in_c = 32;
    let out_c = 64;
    let n_bottlenecks = 1;
    let hidden = out_c / 2;

    let cv1 = make_conv_bn(in_c, 2 * hidden, 1);
    let cat_channels = (2 + n_bottlenecks) * hidden;
    let cv2 = make_conv_bn(cat_channels, out_c, 1);
    let bottlenecks = vec![{
        let b_cv1 = make_conv_bn(hidden, hidden, 3);
        let b_cv2 = make_conv_bn(hidden, hidden, 3);
        Bottleneck::new(b_cv1, b_cv2, false)
    }];

    let c2f = C2f::new(cv1, cv2, bottlenecks);
    let input = DynTensor::full(&[2, 32, 10, 10], 0.5, DType::F32, &Device::Cpu).unwrap();
    let output = c2f.forward(&input).unwrap();
    assert_eq!(output.dims(), &[2, 64, 10, 10]);
}

// ============================================================================
// DocLayout-YOLO integration tests (#3855)
//
// DocLayout-YOLO uses a YOLOv8-based CSPDarknet backbone with 3 C2f stages
// producing P3/P4/P5 features at channels [256, 512, 1024], followed by a
// PAN neck with C2f fusion blocks.
//
// Reference: <https://arxiv.org/abs/2410.12628>
// ============================================================================

/// Helper: build a C2f module from channel counts and bottleneck depth.
fn make_c2f(in_c: usize, out_c: usize, n_bottlenecks: usize, shortcut: bool) -> C2f {
    let hidden = out_c / 2;
    let cv1 = make_conv_bn(in_c, 2 * hidden, 1);
    let cat_channels = (2 + n_bottlenecks) * hidden;
    let cv2 = make_conv_bn(cat_channels, out_c, 1);
    let bottlenecks = (0..n_bottlenecks)
        .map(|_| {
            let b_cv1 = make_conv_bn(hidden, hidden, 3);
            let b_cv2 = make_conv_bn(hidden, hidden, 3);
            Bottleneck::new(b_cv1, b_cv2, shortcut)
        })
        .collect();
    C2f::new(cv1, cv2, bottlenecks)
}

/// DocLayout-YOLO backbone stage 2 (P3): 128 -> 256 channels, depth=3.
/// Production spatial: [B, 128, 80, 80] at stride-8 for 640x640 input.
/// Test uses 20x20 for speed; channel config is the important invariant.
#[test]
fn test_c2f_doclayout_backbone_p3() {
    let c2f = make_c2f(128, 256, 3, true);
    let input = DynTensor::full(&[1, 128, 20, 20], 0.5, DType::F32, &Device::Cpu).unwrap();
    let output = c2f.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 256, 20, 20]);
}

/// DocLayout-YOLO backbone stage 3 (P4): 256 -> 512 channels, depth=3.
/// Production spatial: [B, 256, 40, 40] at stride-16.
#[test]
fn test_c2f_doclayout_backbone_p4() {
    let c2f = make_c2f(256, 512, 3, true);
    let input = DynTensor::full(&[1, 256, 10, 10], 0.5, DType::F32, &Device::Cpu).unwrap();
    let output = c2f.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 512, 10, 10]);
}

/// DocLayout-YOLO backbone stage 4 (P5): 512 -> 1024 channels, depth=3.
/// Production spatial: [B, 512, 20, 20] at stride-32.
#[test]
fn test_c2f_doclayout_backbone_p5() {
    let c2f = make_c2f(512, 1024, 3, true);
    let input = DynTensor::full(&[1, 512, 5, 5], 0.5, DType::F32, &Device::Cpu).unwrap();
    let output = c2f.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 1024, 5, 5]);
}

/// DocLayout-YOLO PAN top-down: concat(P5_up, P4) -> C2f -> N4.
/// Input channels = P5 + P4 = 1024 + 512 = 1536, output = 512, depth=3.
#[test]
fn test_c2f_doclayout_pan_topdown_p5_p4() {
    let c2f = make_c2f(1536, 512, 3, false);
    let input = DynTensor::full(&[1, 1536, 10, 10], 0.5, DType::F32, &Device::Cpu).unwrap();
    let output = c2f.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 512, 10, 10]);
}

/// DocLayout-YOLO PAN top-down: concat(N4_up, P3) -> C2f -> N3.
/// Input channels = N4 + P3 = 512 + 256 = 768, output = 256, depth=3.
#[test]
fn test_c2f_doclayout_pan_topdown_n4_p3() {
    let c2f = make_c2f(768, 256, 3, false);
    let input = DynTensor::full(&[1, 768, 20, 20], 0.5, DType::F32, &Device::Cpu).unwrap();
    let output = c2f.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 256, 20, 20]);
}

/// DocLayout-YOLO PAN bottom-up: concat(N3_down, N4) -> C2f -> N4'.
/// Input channels = N3 + N4 = 256 + 512 = 768, output = 512, depth=3.
#[test]
fn test_c2f_doclayout_pan_bottomup_n3_n4() {
    let c2f = make_c2f(768, 512, 3, false);
    let input = DynTensor::full(&[1, 768, 10, 10], 0.5, DType::F32, &Device::Cpu).unwrap();
    let output = c2f.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 512, 10, 10]);
}

/// DocLayout-YOLO PAN bottom-up: concat(N4'_down, P5) -> C2f -> N5'.
/// Input channels = N4' + P5 = 512 + 1024 = 1536, output = 1024, depth=3.
#[test]
fn test_c2f_doclayout_pan_bottomup_n4_p5() {
    let c2f = make_c2f(1536, 1024, 3, false);
    let input = DynTensor::full(&[1, 1536, 5, 5], 0.5, DType::F32, &Device::Cpu).unwrap();
    let output = c2f.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 1024, 5, 5]);
}

/// Verify C2f internal channel arithmetic for all DocLayout-YOLO configs.
/// hidden = out_c / 2, cat_channels = (2 + depth) * hidden.
#[test]
fn test_c2f_doclayout_channel_arithmetic() {
    let depth = 3_usize;
    // (in_c, out_c) pairs from the DocLayout-YOLO architecture
    let configs: &[(usize, usize)] = &[
        (128, 256),   // backbone P3
        (256, 512),   // backbone P4
        (512, 1024),  // backbone P5
        (1536, 512),  // PAN top-down P5+P4
        (768, 256),   // PAN top-down N4+P3
        (768, 512),   // PAN bottom-up N3+N4
        (1536, 1024), // PAN bottom-up N4'+P5
    ];
    for &(in_c, out_c) in configs {
        let hidden = out_c / 2;
        let cat_channels = (2 + depth) * hidden;
        assert!(
            cat_channels > 0,
            "cat_channels must be positive for ({in_c}, {out_c})"
        );
        assert!(
            hidden > 0,
            "hidden channels must be positive for ({in_c}, {out_c})"
        );
        assert_eq!(
            out_c % 2,
            0,
            "out_c must be even for C2f split at ({in_c}, {out_c})"
        );
    }
}

/// Verify C2f with batch size > 1 using DocLayout-YOLO P3 channel config.
/// Uses reduced spatial dimensions (20x20) for test speed.
#[test]
fn test_c2f_doclayout_batched() {
    let c2f = make_c2f(128, 256, 3, true);
    let input = DynTensor::full(&[4, 128, 20, 20], 0.5, DType::F32, &Device::Cpu).unwrap();
    let output = c2f.forward(&input).unwrap();
    assert_eq!(output.dims(), &[4, 256, 20, 20]);
}
