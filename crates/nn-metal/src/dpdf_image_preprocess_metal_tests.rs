// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for DpdfImagePreprocessMetal GPU dispatch.
//!
//! Verifies shapes, output ranges, and layout conversions for each
//! GPU preprocessing operation using synthetic image data.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;
use nn_models::dpdf_image_preprocess::DpdfPreprocessConfig;

use super::DpdfImagePreprocessMetal;
use crate::test_common::init;

/// Create random f32 data using the test PRNG.
fn rand_f32_vec(seed: u64, count: usize, lo: f32, hi: f32) -> Vec<f32> {
    nn_core::test_prng::rand_f32_vec(seed, count, lo, hi)
}

/// Create a random HWC image tensor `[H, W, 3]`.
fn random_image_hwc(seed: u64, h: usize, w: usize, device: &Device) -> DynTensor {
    let data = rand_f32_vec(seed, h * w * 3, 0.0, 255.0);
    DynTensor::from_vec(data, &[h, w, 3], device).unwrap()
}

/// Create a random CHW image tensor `[3, H, W]`.
fn random_image_chw(seed: u64, h: usize, w: usize, device: &Device) -> DynTensor {
    let data = rand_f32_vec(seed, 3 * h * w, 0.0, 255.0);
    DynTensor::from_vec(data, &[3, h, w], device).unwrap()
}

// ---------------------------------------------------------------------------
// gpu_hwc_to_chw tests
// ---------------------------------------------------------------------------

#[test]
fn test_hwc_to_chw_converts_hwc() {
    init();
    let device = Device::metal();
    let img = random_image_hwc(1, 64, 48, &device);
    assert_eq!(img.dims(), &[64, 48, 3]);

    let chw = DpdfImagePreprocessMetal::gpu_hwc_to_chw(&img).unwrap();
    assert_eq!(chw.dims(), &[3, 64, 48]);
}

#[test]
fn test_hwc_to_chw_passthrough_chw() {
    init();
    let device = Device::metal();
    let img = random_image_chw(2, 64, 48, &device);
    assert_eq!(img.dims(), &[3, 64, 48]);

    let chw = DpdfImagePreprocessMetal::gpu_hwc_to_chw(&img).unwrap();
    assert_eq!(chw.dims(), &[3, 64, 48]);
}

#[test]
fn test_hwc_to_chw_batched_bhwc() {
    init();
    let device = Device::metal();
    let data = rand_f32_vec(3, 1 * 32 * 24 * 3, 0.0, 255.0);
    let img = DynTensor::from_vec(data, &[1, 32, 24, 3], &device).unwrap();

    let bchw = DpdfImagePreprocessMetal::gpu_hwc_to_chw(&img).unwrap();
    assert_eq!(bchw.dims(), &[1, 3, 32, 24]);
}

#[test]
fn test_hwc_to_chw_invalid_rank() {
    init();
    let device = Device::metal();
    let img = DynTensor::from_vec(vec![0.0; 12], &[3, 4], &device).unwrap();
    assert!(DpdfImagePreprocessMetal::gpu_hwc_to_chw(&img).is_err());
}

// ---------------------------------------------------------------------------
// gpu_resize_bilinear tests
// ---------------------------------------------------------------------------

#[test]
fn test_resize_bilinear_downscale() {
    init();
    let device = Device::metal();
    let img = random_image_chw(10, 256, 256, &device);
    let config = DpdfPreprocessConfig::for_granite_docling();
    let pp = DpdfImagePreprocessMetal::new(config);

    let resized = pp.gpu_resize_bilinear(&img, 128, 128).unwrap();
    assert_eq!(resized.dims(), &[3, 128, 128]);
}

#[test]
fn test_resize_bilinear_upscale() {
    init();
    let device = Device::metal();
    let img = random_image_chw(11, 64, 64, &device);
    let config = DpdfPreprocessConfig::for_granite_docling();
    let pp = DpdfImagePreprocessMetal::new(config);

    let resized = pp.gpu_resize_bilinear(&img, 256, 256).unwrap();
    assert_eq!(resized.dims(), &[3, 256, 256]);
}

// ---------------------------------------------------------------------------
// gpu_normalize tests
// ---------------------------------------------------------------------------

#[test]
fn test_normalize_imagenet_range() {
    init();
    let device = Device::metal();
    // Uniform 128.0 image (mid-range uint8 value).
    let data = vec![128.0_f32; 3 * 32 * 32];
    let img = DynTensor::from_vec(data, &[3, 32, 32], &device).unwrap();

    let config = DpdfPreprocessConfig::for_paddle_ocr_detect(); // ImageNet + 1/255 scale
    let pp = DpdfImagePreprocessMetal::new(config);

    let normed = pp
        .gpu_normalize(
            &img,
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )
        .unwrap();
    assert_eq!(normed.dims(), &[3, 32, 32]);

    let cpu = normed.to_device(&Device::Cpu).unwrap();
    let vals = cpu.to_flat_vec::<f32>().unwrap();

    // After scale: 128/255 ~ 0.502
    // R channel: (0.502 - 0.485) / 0.229 ~ 0.074
    let pixels_per_ch = 32 * 32;
    let r_val = vals[0];
    let expected_r = (128.0 / 255.0 - 0.485) / 0.229;
    assert!(
        (r_val - expected_r).abs() < 0.01,
        "R: got {r_val}, expected ~{expected_r}"
    );

    // G channel: (0.502 - 0.456) / 0.224 ~ 0.205
    let g_val = vals[pixels_per_ch];
    let expected_g = (128.0 / 255.0 - 0.456) / 0.224;
    assert!(
        (g_val - expected_g).abs() < 0.01,
        "G: got {g_val}, expected ~{expected_g}"
    );
}

#[test]
fn test_normalize_symmetric_range() {
    init();
    let device = Device::metal();
    // All-zero image -> normalized to (-0.5/0.5) = -1.0 per channel.
    let data = vec![0.0_f32; 3 * 16 * 16];
    let img = DynTensor::from_vec(data, &[3, 16, 16], &device).unwrap();

    let config = DpdfPreprocessConfig::for_granite_docling(); // symmetric [0.5; 3]
    let pp = DpdfImagePreprocessMetal::new(config);

    let normed = pp.gpu_normalize(&img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).unwrap();
    let cpu = normed.to_device(&Device::Cpu).unwrap();
    let vals = cpu.to_flat_vec::<f32>().unwrap();

    // (0 * (1/255) - 0.5) / 0.5 = -1.0
    for v in &vals {
        assert!(
            (*v - (-1.0)).abs() < 1e-4,
            "expected -1.0, got {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// gpu_letterbox_pad tests
// ---------------------------------------------------------------------------

#[test]
fn test_letterbox_pad_square_target() {
    init();
    let device = Device::metal();
    // 64x32 image padded to 64x64 target.
    let img = random_image_chw(20, 64, 32, &device);
    let config = DpdfPreprocessConfig::for_doclayout_yolo();
    let pp = DpdfImagePreprocessMetal::new(config);

    let padded = pp.gpu_letterbox_pad(&img, 64, 64, 0.0).unwrap();
    assert_eq!(padded.dims(), &[3, 64, 64]);
}

#[test]
fn test_letterbox_pad_noop_same_size() {
    init();
    let device = Device::metal();
    let img = random_image_chw(21, 128, 128, &device);
    let config = DpdfPreprocessConfig::for_doclayout_yolo();
    let pp = DpdfImagePreprocessMetal::new(config);

    let padded = pp.gpu_letterbox_pad(&img, 128, 128, 0.0).unwrap();
    assert_eq!(padded.dims(), &[3, 128, 128]);
}

#[test]
fn test_letterbox_pad_error_input_too_large() {
    init();
    let device = Device::metal();
    let img = random_image_chw(22, 256, 256, &device);
    let config = DpdfPreprocessConfig::for_doclayout_yolo();
    let pp = DpdfImagePreprocessMetal::new(config);

    // Input 256x256 is larger than target 128x128.
    assert!(pp.gpu_letterbox_pad(&img, 128, 128, 0.0).is_err());
}

// ---------------------------------------------------------------------------
// Full preprocess_image tests
// ---------------------------------------------------------------------------

#[test]
fn test_preprocess_granite_docling_hwc() {
    init();
    let device = Device::metal();
    let img = random_image_hwc(30, 480, 640, &device);

    let config = DpdfPreprocessConfig::for_granite_docling();
    let pp = DpdfImagePreprocessMetal::new(config.clone());

    let result = pp.preprocess_image(&img).unwrap();
    // Granite Docling: 384x384, no aspect ratio, no padding.
    assert_eq!(result.dims(), &[3, config.target_height as usize, config.target_width as usize]);
}

#[test]
fn test_preprocess_granite_docling_chw() {
    init();
    let device = Device::metal();
    let img = random_image_chw(31, 600, 400, &device);

    let config = DpdfPreprocessConfig::for_granite_docling();
    let pp = DpdfImagePreprocessMetal::new(config);

    let result = pp.preprocess_image(&img).unwrap();
    assert_eq!(result.dims(), &[3, 384, 384]);
}

#[test]
fn test_preprocess_doclayout_yolo_letterbox() {
    init();
    let device = Device::metal();
    // Non-square image with letterbox padding.
    let img = random_image_chw(32, 768, 512, &device);

    let config = DpdfPreprocessConfig::for_doclayout_yolo();
    let pp = DpdfImagePreprocessMetal::new(config);

    let result = pp.preprocess_image(&img).unwrap();
    // DocLayout YOLO: 1024x1024 with letterbox.
    assert_eq!(result.dims(), &[3, 1024, 1024]);
}

#[test]
fn test_preprocess_paddle_ocr_aspect_ratio() {
    init();
    let device = Device::metal();
    // Landscape image with maintain_aspect=true.
    let img = random_image_chw(33, 480, 960, &device);

    let config = DpdfPreprocessConfig::for_paddle_ocr_detect();
    let pp = DpdfImagePreprocessMetal::new(config);

    let result = pp.preprocess_image(&img).unwrap();
    // With maintain_aspect, longer side caps at 960.
    let dims = result.dims();
    assert_eq!(dims[0], 3);
    // Width should be 960, height scaled proportionally.
    assert!(dims[1] <= 960);
    assert!(dims[2] <= 960);
}

#[test]
fn test_preprocess_cpu_input_uploads_to_gpu() {
    init();
    let img = random_image_hwc(34, 100, 100, &Device::Cpu);

    let config = DpdfPreprocessConfig::for_granite_docling();
    let pp = DpdfImagePreprocessMetal::new(config);

    let result = pp.preprocess_image(&img).unwrap();
    assert_eq!(result.dims(), &[3, 384, 384]);
    assert!(result.device().is_gpu(), "result should be on GPU");
}

#[test]
fn test_preprocess_batched_input() {
    init();
    let device = Device::metal();
    let data = rand_f32_vec(35, 1 * 3 * 100 * 100, 0.0, 255.0);
    let img = DynTensor::from_vec(data, &[1, 3, 100, 100], &device).unwrap();

    let config = DpdfPreprocessConfig::for_granite_docling();
    let pp = DpdfImagePreprocessMetal::new(config);

    let result = pp.preprocess_image(&img).unwrap();
    // Batched input should produce batched output.
    assert_eq!(result.dims(), &[1, 3, 384, 384]);
}

#[test]
fn test_preprocess_invalid_rank() {
    init();
    let device = Device::metal();
    let img = DynTensor::from_vec(vec![0.0; 12], &[3, 4], &device).unwrap();

    let config = DpdfPreprocessConfig::for_granite_docling();
    let pp = DpdfImagePreprocessMetal::new(config);

    assert!(pp.preprocess_image(&img).is_err());
}

#[test]
fn test_preprocess_invalid_batch_size() {
    init();
    let device = Device::metal();
    let data = vec![0.0_f32; 2 * 3 * 64 * 64];
    let img = DynTensor::from_vec(data, &[2, 3, 64, 64], &device).unwrap();

    let config = DpdfPreprocessConfig::for_granite_docling();
    let pp = DpdfImagePreprocessMetal::new(config);

    assert!(pp.preprocess_image(&img).is_err());
}

#[test]
fn test_preprocess_output_is_finite() {
    init();
    let device = Device::metal();
    let img = random_image_chw(36, 128, 128, &device);

    let config = DpdfPreprocessConfig::for_granite_docling();
    let pp = DpdfImagePreprocessMetal::new(config);

    let result = pp.preprocess_image(&img).unwrap();
    let cpu = result.to_device(&Device::Cpu).unwrap();
    let vals = cpu.to_flat_vec::<f32>().unwrap();

    for (i, v) in vals.iter().enumerate() {
        assert!(
            v.is_finite(),
            "non-finite value at index {i}: {v}"
        );
    }
}
