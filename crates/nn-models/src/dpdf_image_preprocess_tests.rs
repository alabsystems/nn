// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for dpdf image preprocessing pipeline.

use super::*;

// ---------------------------------------------------------------------------
// Preset config value tests
// ---------------------------------------------------------------------------

#[test]
fn test_granite_docling_preset_config_values() {
    let cfg = DpdfPreprocessConfig::for_granite_docling();
    assert_eq!(cfg.target_height, 384);
    assert_eq!(cfg.target_width, 384);
    assert_eq!(cfg.mean, [0.5, 0.5, 0.5]);
    assert_eq!(cfg.std, [0.5, 0.5, 0.5]);
    assert_eq!(cfg.padding_mode, PaddingMode::None);
    assert!((cfg.scale_factor - 1.0 / 255.0).abs() < 1e-7);
    assert!(!cfg.maintain_aspect);
}

#[test]
fn test_doclayout_yolo_preset_config_values() {
    let cfg = DpdfPreprocessConfig::for_doclayout_yolo();
    assert_eq!(cfg.target_height, 1024);
    assert_eq!(cfg.target_width, 1024);
    assert_eq!(cfg.mean, [0.0, 0.0, 0.0]);
    assert_eq!(cfg.std, [1.0, 1.0, 1.0]);
    assert_eq!(
        cfg.padding_mode,
        PaddingMode::Letterbox { fill_value: 114.0 }
    );
    assert!(cfg.maintain_aspect);
}

#[test]
fn test_paddle_ocr_detect_preset_config_values() {
    let cfg = DpdfPreprocessConfig::for_paddle_ocr_detect();
    assert_eq!(cfg.target_height, 960);
    assert_eq!(cfg.target_width, 960);
    assert_eq!(cfg.mean, [0.485, 0.456, 0.406]);
    assert_eq!(cfg.std, [0.229, 0.224, 0.225]);
    assert_eq!(cfg.padding_mode, PaddingMode::None);
    assert!(cfg.maintain_aspect);
}

#[test]
fn test_paddle_ocr_recognize_preset_config_values() {
    let cfg = DpdfPreprocessConfig::for_paddle_ocr_recognize();
    assert_eq!(cfg.target_height, 48);
    assert_eq!(cfg.target_width, 320);
    assert_eq!(cfg.mean, [0.485, 0.456, 0.406]);
    assert_eq!(cfg.std, [0.229, 0.224, 0.225]);
    assert!(cfg.maintain_aspect);
}

#[test]
fn test_table_transformer_preset_config_values() {
    let cfg = DpdfPreprocessConfig::for_table_transformer();
    assert_eq!(cfg.target_height, 800);
    assert_eq!(cfg.target_width, 800);
    assert_eq!(cfg.mean, [0.485, 0.456, 0.406]);
    assert_eq!(cfg.std, [0.229, 0.224, 0.225]);
    assert!(cfg.maintain_aspect);
}

#[test]
fn test_qwen3_vl_preset_config_values() {
    let cfg = DpdfPreprocessConfig::for_qwen3_vl();
    assert_eq!(cfg.target_height, 0);
    assert_eq!(cfg.target_width, 0);
    assert_eq!(cfg.mean, [0.5, 0.5, 0.5]);
    assert_eq!(cfg.std, [0.5, 0.5, 0.5]);
    assert_eq!(cfg.min_pixels, 256 * 28 * 28);
    assert_eq!(cfg.max_pixels, 1280 * 28 * 28);
    assert_eq!(cfg.patch_size, 28);
    assert!(cfg.maintain_aspect);
}

#[test]
fn test_glm_ocr_preset_config_values() {
    let cfg = DpdfPreprocessConfig::for_glm_ocr();
    assert_eq!(cfg.target_height, 1120);
    assert_eq!(cfg.target_width, 1120);
    assert_eq!(cfg.mean, [0.5, 0.5, 0.5]);
    assert_eq!(cfg.std, [0.5, 0.5, 0.5]);
    assert!(cfg.maintain_aspect);
}

// ---------------------------------------------------------------------------
// Letterbox computation tests
// ---------------------------------------------------------------------------

#[test]
fn test_letterbox_params_square_image_in_square_target() {
    let params = compute_letterbox_params(100, 100, 100, 100);
    assert_eq!(params.top, 0);
    assert_eq!(params.left, 0);
    assert_eq!(params.bottom, 0);
    assert_eq!(params.right, 0);
}

#[test]
fn test_letterbox_params_wide_image_centered() {
    // 200x400 image in 400x400 target → 100px padding top and bottom.
    let params = compute_letterbox_params(200, 400, 400, 400);
    assert_eq!(params.top, 100);
    assert_eq!(params.bottom, 100);
    assert_eq!(params.left, 0);
    assert_eq!(params.right, 0);
}

#[test]
fn test_letterbox_params_tall_image_centered() {
    // 400x200 image in 400x400 target → 100px padding left and right.
    let params = compute_letterbox_params(400, 200, 400, 400);
    assert_eq!(params.top, 0);
    assert_eq!(params.bottom, 0);
    assert_eq!(params.left, 100);
    assert_eq!(params.right, 100);
}

#[test]
fn test_letterbox_params_odd_padding_split() {
    // 99x100 in 100x100 → 1px total vertical pad, top=0, bottom=1.
    let params = compute_letterbox_params(99, 100, 100, 100);
    assert_eq!(params.top, 0);
    assert_eq!(params.bottom, 1);
    assert_eq!(params.left, 0);
    assert_eq!(params.right, 0);
}

// ---------------------------------------------------------------------------
// Resize dimension computation tests
// ---------------------------------------------------------------------------

#[test]
fn test_resize_dims_no_maintain_aspect() {
    let (h, w) = compute_resize_dims(480, 640, 224, 224, false);
    assert_eq!(h, 224);
    assert_eq!(w, 224);
}

#[test]
fn test_resize_dims_maintain_aspect_landscape() {
    // 480x640 → target 1024x1024, scale = 1024/640 = 1.6.
    // h = 480 * 1.6 = 768, w = 640 * 1.6 = 1024.
    let (h, w) = compute_resize_dims(480, 640, 1024, 1024, true);
    assert_eq!(w, 1024);
    assert_eq!(h, 768);
}

#[test]
fn test_resize_dims_maintain_aspect_portrait() {
    // 800x600 → target 400x400, scale = 400/800 = 0.5.
    // h = 800 * 0.5 = 400, w = 600 * 0.5 = 300.
    let (h, w) = compute_resize_dims(800, 600, 400, 400, true);
    assert_eq!(h, 400);
    assert_eq!(w, 300);
}

#[test]
fn test_resize_dims_already_at_target() {
    let (h, w) = compute_resize_dims(384, 384, 384, 384, true);
    assert_eq!(h, 384);
    assert_eq!(w, 384);
}

// ---------------------------------------------------------------------------
// Normalization math tests
// ---------------------------------------------------------------------------

#[test]
fn test_preprocess_normalization_identity() {
    // 2x2 image, all pixels = 128.0, scale=1/255, mean=0, std=1.
    let pixels: Vec<f32> = vec![128.0; 2 * 2 * 3];
    let cfg = DpdfPreprocessConfig {
        target_height: 2,
        target_width: 2,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
        padding_mode: PaddingMode::None,
        scale_factor: 1.0 / 255.0,
        maintain_aspect: false,
        min_pixels: 0,
        max_pixels: 0,
        patch_size: 0,
    };

    let result = preprocess(&pixels, 2, 2, &cfg).unwrap();
    let expected = 128.0 / 255.0;
    for &v in &result.data {
        assert!((v - expected).abs() < 1e-5, "expected {expected}, got {v}");
    }
    assert_eq!(result.height, 2);
    assert_eq!(result.width, 2);
    assert_eq!(result.channels, 3);
}

#[test]
fn test_preprocess_normalization_symmetric() {
    // All pixels = 255.0, scale=1/255, mean=0.5, std=0.5.
    // Result: (1.0 - 0.5) / 0.5 = 1.0.
    let pixels: Vec<f32> = vec![255.0; 2 * 2 * 3];
    let cfg = DpdfPreprocessConfig {
        target_height: 2,
        target_width: 2,
        mean: SYMMETRIC_MEAN,
        std: SYMMETRIC_STD,
        padding_mode: PaddingMode::None,
        scale_factor: 1.0 / 255.0,
        maintain_aspect: false,
        min_pixels: 0,
        max_pixels: 0,
        patch_size: 0,
    };

    let result = preprocess(&pixels, 2, 2, &cfg).unwrap();
    for &v in &result.data {
        assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
    }
}

#[test]
fn test_preprocess_chw_layout() {
    // 1x2 image: pixel(0,0) = [10, 20, 30], pixel(0,1) = [40, 50, 60].
    // No normalization (mean=0, std=1, scale=1).
    let pixels: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let cfg = DpdfPreprocessConfig {
        target_height: 1,
        target_width: 2,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
        padding_mode: PaddingMode::None,
        scale_factor: 1.0,
        maintain_aspect: false,
        min_pixels: 0,
        max_pixels: 0,
        patch_size: 0,
    };

    let result = preprocess(&pixels, 1, 2, &cfg).unwrap();
    // CHW: C0=[10, 40], C1=[20, 50], C2=[30, 60].
    assert_eq!(result.data.len(), 6);
    assert!((result.data[0] - 10.0).abs() < 1e-5); // C0, pixel 0
    assert!((result.data[1] - 40.0).abs() < 1e-5); // C0, pixel 1
    assert!((result.data[2] - 20.0).abs() < 1e-5); // C1, pixel 0
    assert!((result.data[3] - 50.0).abs() < 1e-5); // C1, pixel 1
    assert!((result.data[4] - 30.0).abs() < 1e-5); // C2, pixel 0
    assert!((result.data[5] - 60.0).abs() < 1e-5); // C2, pixel 1
}

// ---------------------------------------------------------------------------
// Edge case / error tests
// ---------------------------------------------------------------------------

#[test]
fn test_preprocess_zero_height_returns_none() {
    let cfg = DpdfPreprocessConfig::for_granite_docling();
    assert!(preprocess(&[], 0, 100, &cfg).is_none());
}

#[test]
fn test_preprocess_zero_width_returns_none() {
    let cfg = DpdfPreprocessConfig::for_granite_docling();
    assert!(preprocess(&[], 100, 0, &cfg).is_none());
}

#[test]
fn test_preprocess_short_buffer_returns_none() {
    let pixels: Vec<f32> = vec![0.0; 5]; // too short for 2x2x3=12
    let cfg = DpdfPreprocessConfig::for_granite_docling();
    assert!(preprocess(&pixels, 2, 2, &cfg).is_none());
}

#[test]
fn test_preprocess_letterbox_output_dimensions() {
    // 300x400 source, YOLO 1024x1024 letterbox.
    let pixels: Vec<f32> = vec![128.0; 300 * 400 * 3];
    let cfg = DpdfPreprocessConfig::for_doclayout_yolo();
    let result = preprocess(&pixels, 300, 400, &cfg).unwrap();
    assert_eq!(result.height, 1024);
    assert_eq!(result.width, 1024);
    // Data length = 3 * 1024 * 1024.
    assert_eq!(result.data.len(), 3 * 1024 * 1024);
}

#[test]
fn test_preprocess_aspect_ratio_preserved() {
    // 100x200 source, target 400x400, maintain_aspect=true → 200x400.
    let (h, w) = compute_resize_dims(100, 200, 400, 400, true);
    // Scale = min(400/100, 400/200) = min(4, 2) = 2.
    // h = 100*2 = 200, w = 200*2 = 400.
    assert_eq!(h, 200);
    assert_eq!(w, 400);
}

#[test]
fn test_preprocess_center_crop_output_dimensions() {
    let pixels: Vec<f32> = vec![128.0; 100 * 200 * 3];
    let cfg = DpdfPreprocessConfig {
        target_height: 50,
        target_width: 50,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
        padding_mode: PaddingMode::CenterCrop,
        scale_factor: 1.0 / 255.0,
        maintain_aspect: false,
        min_pixels: 0,
        max_pixels: 0,
        patch_size: 0,
    };
    let result = preprocess(&pixels, 100, 200, &cfg).unwrap();
    assert_eq!(result.height, 50);
    assert_eq!(result.width, 50);
    assert_eq!(result.data.len(), 3 * 50 * 50);
}
