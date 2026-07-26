// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`ImageProcessor`].

use super::*;
use crate::Device;

/// Verify ImageNet preset has correct mean/std.
#[test]
fn test_image_processor_imagenet_preset() {
    let proc = ImageProcessor::imagenet(224);
    assert_eq!(proc.target_height(), 224);
    assert_eq!(proc.target_width(), 224);
    assert_eq!(proc.mean(), &[0.485, 0.456, 0.406]);
    assert_eq!(proc.std_dev(), &[0.229, 0.224, 0.225]);
}

/// Verify SigLIP2 preset has correct mean/std.
#[test]
fn test_image_processor_siglip2_preset() {
    let proc = ImageProcessor::siglip2(384);
    assert_eq!(proc.target_height(), 384);
    assert_eq!(proc.target_width(), 384);
    assert_eq!(proc.mean(), &[0.5, 0.5, 0.5]);
    assert_eq!(proc.std_dev(), &[0.5, 0.5, 0.5]);
}

/// Verify normalize-only preset has identity normalization.
#[test]
fn test_image_processor_normalize_only_preset() {
    let proc = ImageProcessor::normalize_only(256, 128);
    assert_eq!(proc.target_height(), 256);
    assert_eq!(proc.target_width(), 128);
    assert_eq!(proc.mean(), &[0.0, 0.0, 0.0]);
    assert_eq!(proc.std_dev(), &[1.0, 1.0, 1.0]);
}

/// Process a 4x4 RGB image and verify output shape is [1, 3, H, W].
#[test]
fn test_image_processor_output_shape() {
    let proc = ImageProcessor::imagenet(224);
    let pixels = vec![128u8; 4 * 4 * 3];
    let tensor = proc.process(&pixels, 4, 4, 3, &Device::Cpu).unwrap();
    assert_eq!(tensor.dims(), &[1, 3, 224, 224]);
}

/// Process a 4x4 image with normalize_only and verify pixel values.
#[test]
fn test_image_processor_normalize_only_values() {
    // Use normalize_only at 4x4 (no resize) to check [0,1] conversion.
    let proc = ImageProcessor::normalize_only(4, 4);
    // All pixels = 255 -> should normalize to 1.0
    let pixels = vec![255u8; 4 * 4 * 3];
    let tensor = proc.process(&pixels, 4, 4, 3, &Device::Cpu).unwrap();
    let data = tensor.to_flat_vec::<f32>().unwrap();
    // All values should be (255/255 - 0) / 1 = 1.0
    for &v in &data {
        assert!((v - 1.0).abs() < 1e-6, "expected 1.0, got {v}");
    }
}

/// Verify HWC -> CHW conversion is correct.
#[test]
fn test_image_processor_hwc_to_chw() {
    // Create a 2x2 image with distinct per-channel values.
    // Pixel layout (HWC): R G B for each of 4 pixels.
    let proc = ImageProcessor::normalize_only(2, 2);
    let pixels: Vec<u8> = vec![
        255, 0, 0, // (0,0) = red
        0, 255, 0, // (0,1) = green
        0, 0, 255, // (1,0) = blue
        128, 128, 128, // (1,1) = gray
    ];
    let tensor = proc.process(&pixels, 2, 2, 3, &Device::Cpu).unwrap();
    assert_eq!(tensor.dims(), &[1, 3, 2, 2]);
    let data = tensor.to_flat_vec::<f32>().unwrap();

    // CHW layout: channel 0 (R) = [1.0, 0.0, 0.0, 128/255]
    let r_channel = &data[0..4];
    assert!(
        (r_channel[0] - 1.0).abs() < 1e-5,
        "R(0,0) = {}",
        r_channel[0]
    );
    assert!(
        (r_channel[1] - 0.0).abs() < 1e-5,
        "R(0,1) = {}",
        r_channel[1]
    );
    assert!(
        (r_channel[2] - 0.0).abs() < 1e-5,
        "R(1,0) = {}",
        r_channel[2]
    );
    let gray = 128.0 / 255.0;
    assert!(
        (r_channel[3] - gray).abs() < 1e-5,
        "R(1,1) = {}",
        r_channel[3]
    );

    // Channel 1 (G) = [0.0, 1.0, 0.0, 128/255]
    let g_channel = &data[4..8];
    assert!((g_channel[0] - 0.0).abs() < 1e-5);
    assert!((g_channel[1] - 1.0).abs() < 1e-5);
    assert!((g_channel[2] - 0.0).abs() < 1e-5);
    assert!((g_channel[3] - gray).abs() < 1e-5);

    // Channel 2 (B) = [0.0, 0.0, 1.0, 128/255]
    let b_channel = &data[8..12];
    assert!((b_channel[0] - 0.0).abs() < 1e-5);
    assert!((b_channel[1] - 0.0).abs() < 1e-5);
    assert!((b_channel[2] - 1.0).abs() < 1e-5);
    assert!((b_channel[3] - gray).abs() < 1e-5);
}

/// Test ImageNet normalization produces expected values.
#[test]
fn test_image_processor_imagenet_normalization() {
    // 2x2 solid gray (128) image, processed at 2x2 (no resize).
    let proc = ImageProcessor::new(2, 2, IMAGENET_MEAN, IMAGENET_STD);
    let pixels = vec![128u8; 2 * 2 * 3];
    let tensor = proc.process(&pixels, 2, 2, 3, &Device::Cpu).unwrap();
    let data = tensor.to_flat_vec::<f32>().unwrap();

    let pixel_val = 128.0 / 255.0;
    // R channel: (pixel_val - 0.485) / 0.229
    let expected_r = (pixel_val - 0.485) / 0.229;
    assert!(
        (data[0] - expected_r).abs() < 1e-5,
        "expected {expected_r}, got {}",
        data[0]
    );
    // G channel: (pixel_val - 0.456) / 0.224
    let expected_g = (pixel_val - 0.456) / 0.224;
    assert!(
        (data[4] - expected_g).abs() < 1e-5,
        "expected {expected_g}, got {}",
        data[4]
    );
    // B channel: (pixel_val - 0.406) / 0.225
    let expected_b = (pixel_val - 0.406) / 0.225;
    assert!(
        (data[8] - expected_b).abs() < 1e-5,
        "expected {expected_b}, got {}",
        data[8]
    );
}

/// Test bilinear resize from 8x8 -> 4x4.
#[test]
fn test_image_processor_bilinear_resize() {
    // Create an 8x8 solid white image, resize to 4x4.
    let proc = ImageProcessor::normalize_only(4, 4);
    let pixels = vec![200u8; 8 * 8 * 3];
    let tensor = proc.process(&pixels, 8, 8, 3, &Device::Cpu).unwrap();
    assert_eq!(tensor.dims(), &[1, 3, 4, 4]);
    let data = tensor.to_flat_vec::<f32>().unwrap();
    // All pixels were 200 -> after bilinear resize all should still be ~200/255.
    let expected = 200.0 / 255.0;
    for (i, &v) in data.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-4,
            "pixel {i}: expected {expected}, got {v}"
        );
    }
}

/// Test bilinear resize produces reasonable interpolated values for a gradient.
#[test]
fn test_image_processor_bilinear_resize_gradient() {
    // Create a 4x4 image with horizontal gradient in R channel.
    // R increases left to right: col 0=0, col 1=85, col 2=170, col 3=255.
    let mut pixels = vec![0u8; 4 * 4 * 3];
    for row in 0..4 {
        for col in 0..4 {
            let idx = (row * 4 + col) * 3;
            pixels[idx] = (col as u8) * 85; // R
            pixels[idx + 1] = 0; // G
            pixels[idx + 2] = 0; // B
        }
    }

    // Resize 4x4 -> 2x2. The center-aligned bilinear sampling should produce
    // interpolated values somewhere between the extremes.
    let proc = ImageProcessor::normalize_only(2, 2);
    let tensor = proc.process(&pixels, 4, 4, 3, &Device::Cpu).unwrap();
    assert_eq!(tensor.dims(), &[1, 3, 2, 2]);
    let data = tensor.to_flat_vec::<f32>().unwrap();

    // R channel: data[0..4] for the 2x2 output.
    // Left column should be smaller than right column.
    let r_00 = data[0]; // top-left
    let r_01 = data[1]; // top-right
    assert!(
        r_00 < r_01,
        "horizontal gradient preserved: R(0,0)={r_00} < R(0,1)={r_01}"
    );
}

/// Test that zero-dimension inputs are rejected.
#[test]
fn test_image_processor_zero_dimensions() {
    let proc = ImageProcessor::imagenet(224);
    let result = proc.process(&[], 0, 4, 3, &Device::Cpu);
    assert!(result.is_err());
    let result = proc.process(&[], 4, 0, 3, &Device::Cpu);
    assert!(result.is_err());
}

/// Test that non-RGB channels are rejected.
#[test]
fn test_image_processor_wrong_channels() {
    let proc = ImageProcessor::imagenet(224);
    let pixels = vec![128u8; 4 * 4 * 4]; // RGBA
    let result = proc.process(&pixels, 4, 4, 4, &Device::Cpu);
    assert!(result.is_err());
}

/// Test that too-short pixel buffer is rejected.
#[test]
fn test_image_processor_short_buffer() {
    let proc = ImageProcessor::imagenet(224);
    let pixels = vec![128u8; 10]; // way too short for 4x4x3
    let result = proc.process(&pixels, 4, 4, 3, &Device::Cpu);
    assert!(result.is_err());
}

/// Test process_tensor with [H, W, 3] input.
#[test]
fn test_image_processor_process_tensor_rank3() {
    let proc = ImageProcessor::normalize_only(2, 2);
    // Create a [4, 4, 3] float tensor with all 0.5.
    let data = vec![0.5f32; 4 * 4 * 3];
    let input = DynTensor::from_vec(data, &[4, 4, 3], &Device::Cpu).unwrap();
    let output = proc.process_tensor(&input, &Device::Cpu).unwrap();
    assert_eq!(output.dims(), &[1, 3, 2, 2]);

    let out_data = output.to_flat_vec::<f32>().unwrap();
    for &v in &out_data {
        assert!((v - 0.5).abs() < 1e-4, "expected ~0.5, got {v}");
    }
}

/// Test process_tensor with [B, H, W, 3] input (batch=2).
#[test]
fn test_image_processor_process_tensor_batched() {
    let proc = ImageProcessor::normalize_only(2, 2);
    // Batch of 2 images, 3x3 each, all 0.8.
    let data = vec![0.8f32; 2 * 3 * 3 * 3];
    let input = DynTensor::from_vec(data, &[2, 3, 3, 3], &Device::Cpu).unwrap();
    let output = proc.process_tensor(&input, &Device::Cpu).unwrap();
    assert_eq!(output.dims(), &[2, 3, 2, 2]);
}

/// Test the standalone bilinear_resize_f32 function for identity (same size).
#[test]
fn test_bilinear_resize_identity() {
    let src: Vec<f32> = (0..12).map(|i| i as f32).collect(); // 2x2x3
    let dst = bilinear_resize_f32(&src, 2, 2, 2, 2, 3);
    for (i, (&s, &d)) in src.iter().zip(dst.iter()).enumerate() {
        assert!((s - d).abs() < 1e-5, "pixel {i}: src={s}, dst={d}");
    }
}
