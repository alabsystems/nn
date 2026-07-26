// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`ImagePreprocessor`].

use super::*;
use crate::Device;

// ---------------------------------------------------------------------------
// Factory preset tests
// ---------------------------------------------------------------------------

#[test]
fn test_siglip2_preset_config() {
    let p = ImagePreprocessor::siglip2();
    assert_eq!(p.target_height(), 384);
    assert_eq!(p.target_width(), 384);
    assert_eq!(p.mean(), &[0.5, 0.5, 0.5]);
    assert_eq!(p.std_dev(), &[0.5, 0.5, 0.5]);
    assert!((p.rescale_factor() - 1.0 / 255.0).abs() < 1e-8);
}

#[test]
fn test_vit_base_preset_config() {
    let p = ImagePreprocessor::vit_base();
    assert_eq!(p.target_height(), 224);
    assert_eq!(p.target_width(), 224);
    assert_eq!(p.mean(), &[0.485, 0.456, 0.406]);
    assert_eq!(p.std_dev(), &[0.229, 0.224, 0.225]);
    assert!((p.rescale_factor() - 1.0 / 255.0).abs() < 1e-8);
}

#[test]
fn test_qwen_vl_preset_config() {
    let p = ImagePreprocessor::qwen_vl();
    assert_eq!(p.target_height(), 448);
    assert_eq!(p.target_width(), 448);
    // Uses ImageNet constants.
    assert_eq!(p.mean(), &[0.485, 0.456, 0.406]);
    assert_eq!(p.std_dev(), &[0.229, 0.224, 0.225]);
}

// ---------------------------------------------------------------------------
// Constructor validation
// ---------------------------------------------------------------------------

#[test]
fn test_zero_std_rejected() {
    let result = ImagePreprocessor::new(224, 224, [0.5; 3], [0.0, 0.5, 0.5], 1.0 / 255.0);
    assert!(result.is_err(), "std[0]=0.0 should be rejected");
}

#[test]
fn test_zero_std_channel2_rejected() {
    let result = ImagePreprocessor::new(224, 224, [0.5; 3], [0.5, 0.5, 0.0], 1.0 / 255.0);
    assert!(result.is_err(), "std[2]=0.0 should be rejected");
}

#[test]
fn test_valid_custom_construction() {
    let p = ImagePreprocessor::new(256, 128, [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], 1.0).unwrap();
    assert_eq!(p.target_height(), 256);
    assert_eq!(p.target_width(), 128);
    assert!((p.rescale_factor() - 1.0).abs() < 1e-8);
}

// ---------------------------------------------------------------------------
// Preprocess: rank-3 CHW input
// ---------------------------------------------------------------------------

#[test]
fn test_preprocess_chw_shape_preserved() {
    let p = ImagePreprocessor::vit_base();
    let image = DynTensor::full(&[3, 224, 224], 128.0, crate::DType::F32, &Device::Cpu).unwrap();
    let out = p.preprocess(&image).unwrap();
    assert_eq!(out.dims(), &[3, 224, 224]);
}

#[test]
fn test_preprocess_chw_rescale_and_normalize() {
    // Use identity-like normalization to isolate rescale.
    let p = ImagePreprocessor::new(2, 2, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0 / 255.0).unwrap();
    // All pixels = 255.0 -> rescaled = 1.0, normalized = (1.0 - 0) / 1 = 1.0.
    let image = DynTensor::full(&[3, 2, 2], 255.0, crate::DType::F32, &Device::Cpu).unwrap();
    let out = p.preprocess(&image).unwrap();
    let data = out.to_flat_vec::<f32>().unwrap();
    for (i, &v) in data.iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-5, "pixel {i}: expected 1.0, got {v}");
    }
}

#[test]
fn test_preprocess_chw_imagenet_normalization() {
    // Use custom preprocessor at 2x2 to avoid resize changing values.
    let p = ImagePreprocessor::new(2, 2, IMAGENET_MEAN, IMAGENET_STD, 1.0 / 255.0).unwrap();
    // Solid 128 image (3, 2, 2).
    let image = DynTensor::full(&[3, 2, 2], 128.0, crate::DType::F32, &Device::Cpu).unwrap();
    let out = p.preprocess(&image).unwrap();
    let data = out.to_flat_vec::<f32>().unwrap();

    let pixel_val = 128.0 / 255.0;
    // R channel first 4 values: (pixel_val - 0.485) / 0.229
    let expected_r = (pixel_val - 0.485) / 0.229;
    for i in 0..4 {
        assert!(
            (data[i] - expected_r).abs() < 1e-4,
            "R[{i}]: expected {expected_r}, got {}",
            data[i]
        );
    }
    // G channel: values [4..8]
    let expected_g = (pixel_val - 0.456) / 0.224;
    for i in 4..8 {
        assert!(
            (data[i] - expected_g).abs() < 1e-4,
            "G[{i}]: expected {expected_g}, got {}",
            data[i]
        );
    }
    // B channel: values [8..12]
    let expected_b = (pixel_val - 0.406) / 0.225;
    for i in 8..12 {
        assert!(
            (data[i] - expected_b).abs() < 1e-4,
            "B[{i}]: expected {expected_b}, got {}",
            data[i]
        );
    }
}

#[test]
fn test_preprocess_siglip2_range() {
    let p = ImagePreprocessor::siglip2();
    // All-zero pixels -> (0 * 1/255 - 0.5) / 0.5 = -1.0
    let image = DynTensor::full(&[3, 4, 4], 0.0, crate::DType::F32, &Device::Cpu).unwrap();
    let out = p.preprocess(&image).unwrap();
    let data = out.to_flat_vec::<f32>().unwrap();
    for (i, &v) in data.iter().enumerate() {
        assert!(
            (v - (-1.0)).abs() < 1e-5,
            "pixel {i}: expected -1.0, got {v}"
        );
    }

    // All 255 pixels -> (255 * 1/255 - 0.5) / 0.5 = 1.0
    let image = DynTensor::full(&[3, 4, 4], 255.0, crate::DType::F32, &Device::Cpu).unwrap();
    let out = p.preprocess(&image).unwrap();
    let data = out.to_flat_vec::<f32>().unwrap();
    for (i, &v) in data.iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-5, "pixel {i}: expected 1.0, got {v}");
    }
}

// ---------------------------------------------------------------------------
// Preprocess: rank-4 BCHW input
// ---------------------------------------------------------------------------

#[test]
fn test_preprocess_bchw_shape_preserved() {
    let p = ImagePreprocessor::vit_base();
    let image = DynTensor::full(&[2, 3, 224, 224], 128.0, crate::DType::F32, &Device::Cpu).unwrap();
    let out = p.preprocess(&image).unwrap();
    assert_eq!(out.dims(), &[2, 3, 224, 224]);
}

#[test]
fn test_preprocess_bchw_values() {
    let p = ImagePreprocessor::new(2, 2, [0.0; 3], [1.0; 3], 0.5).unwrap();
    // Batch of 2, all 2.0 -> rescaled = 1.0, normalized = (1.0 - 0) / 1 = 1.0.
    let image = DynTensor::full(&[2, 3, 2, 2], 2.0, crate::DType::F32, &Device::Cpu).unwrap();
    let out = p.preprocess(&image).unwrap();
    let data = out.to_flat_vec::<f32>().unwrap();
    for (i, &v) in data.iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-5, "pixel {i}: expected 1.0, got {v}");
    }
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

#[test]
fn test_preprocess_wrong_rank_rejected() {
    let p = ImagePreprocessor::vit_base();
    // Rank 2 - neither CHW nor BCHW.
    let image = DynTensor::full(&[3, 224], 128.0, crate::DType::F32, &Device::Cpu).unwrap();
    assert!(p.preprocess(&image).is_err());
}

#[test]
fn test_preprocess_rank5_rejected() {
    let p = ImagePreprocessor::vit_base();
    let image = DynTensor::full(&[1, 1, 3, 2, 2], 128.0, crate::DType::F32, &Device::Cpu).unwrap();
    assert!(p.preprocess(&image).is_err());
}

#[test]
fn test_preprocess_wrong_channels_chw() {
    let p = ImagePreprocessor::vit_base();
    // 4 channels instead of 3.
    let image = DynTensor::full(&[4, 224, 224], 128.0, crate::DType::F32, &Device::Cpu).unwrap();
    assert!(p.preprocess(&image).is_err());
}

#[test]
fn test_preprocess_wrong_channels_bchw() {
    let p = ImagePreprocessor::vit_base();
    // 1 channel instead of 3.
    let image = DynTensor::full(&[2, 1, 224, 224], 128.0, crate::DType::F32, &Device::Cpu).unwrap();
    assert!(p.preprocess(&image).is_err());
}

// ---------------------------------------------------------------------------
// Per-channel distinct values
// ---------------------------------------------------------------------------

#[test]
fn test_preprocess_per_channel_normalization() {
    // Verify each channel uses its own mean/std.
    let mean = [0.1, 0.2, 0.3];
    let std = [0.4, 0.5, 0.6];
    let p = ImagePreprocessor::new(1, 1, mean, std, 1.0).unwrap();

    // Input: [3, 1, 1] with values [1.0, 2.0, 3.0].
    let image = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1, 1], &Device::Cpu).unwrap();
    let out = p.preprocess(&image).unwrap();
    let data = out.to_flat_vec::<f32>().unwrap();

    // rescale_factor=1.0, so rescale is identity.
    // Channel 0: (1.0 - 0.1) / 0.4 = 2.25
    assert!((data[0] - 2.25).abs() < 1e-5, "ch0: got {}", data[0]);
    // Channel 1: (2.0 - 0.2) / 0.5 = 3.6
    assert!((data[1] - 3.6).abs() < 1e-5, "ch1: got {}", data[1]);
    // Channel 2: (3.0 - 0.3) / 0.6 = 4.5
    assert!((data[2] - 4.5).abs() < 1e-5, "ch2: got {}", data[2]);
}

// ---------------------------------------------------------------------------
// Bilinear resize integration (#3531)
// ---------------------------------------------------------------------------

/// Resize from [3, 8, 8] to [3, 4, 4]. Solid-color input preserves values through resize.
#[test]
fn test_preprocess_resize_downscale_solid() {
    let p = ImagePreprocessor::new(4, 4, [0.0; 3], [1.0; 3], 1.0).unwrap();
    let image = DynTensor::full(&[3, 8, 8], 0.75, crate::DType::F32, &Device::Cpu).unwrap();
    let out = p.preprocess(&image).unwrap();
    assert_eq!(out.dims(), &[3, 4, 4]);
    let data = out.to_flat_vec::<f32>().unwrap();
    for (i, &v) in data.iter().enumerate() {
        assert!((v - 0.75).abs() < 1e-4, "pixel {i}: expected 0.75, got {v}");
    }
}

/// Resize from [3, 2, 2] to [3, 4, 4] (upscale). Solid-color preserved.
#[test]
fn test_preprocess_resize_upscale_solid() {
    let p = ImagePreprocessor::new(4, 4, [0.0; 3], [1.0; 3], 1.0).unwrap();
    let image = DynTensor::full(&[3, 2, 2], 0.5, crate::DType::F32, &Device::Cpu).unwrap();
    let out = p.preprocess(&image).unwrap();
    assert_eq!(out.dims(), &[3, 4, 4]);
    let data = out.to_flat_vec::<f32>().unwrap();
    for (i, &v) in data.iter().enumerate() {
        assert!((v - 0.5).abs() < 1e-4, "pixel {i}: expected 0.5, got {v}");
    }
}

/// Non-square resize: [3, 10, 20] -> [3, 5, 5].
#[test]
fn test_preprocess_resize_non_square() {
    let p = ImagePreprocessor::new(5, 5, [0.0; 3], [1.0; 3], 1.0).unwrap();
    let image = DynTensor::full(&[3, 10, 20], 1.0, crate::DType::F32, &Device::Cpu).unwrap();
    let out = p.preprocess(&image).unwrap();
    assert_eq!(out.dims(), &[3, 5, 5]);
}

/// Batched resize: [2, 3, 8, 8] -> [2, 3, 4, 4].
#[test]
fn test_preprocess_resize_batched() {
    let p = ImagePreprocessor::new(4, 4, [0.0; 3], [1.0; 3], 1.0).unwrap();
    let image = DynTensor::full(&[2, 3, 8, 8], 0.25, crate::DType::F32, &Device::Cpu).unwrap();
    let out = p.preprocess(&image).unwrap();
    assert_eq!(out.dims(), &[2, 3, 4, 4]);
    let data = out.to_flat_vec::<f32>().unwrap();
    for (i, &v) in data.iter().enumerate() {
        assert!((v - 0.25).abs() < 1e-4, "pixel {i}: expected 0.25, got {v}");
    }
}

/// Full SigLIP2 end-to-end: [3, 640, 480] -> resize 384x384 -> rescale -> normalize.
/// Solid 255.0: rescaled = 1.0, normalized = (1.0 - 0.5) / 0.5 = 1.0.
#[test]
fn test_preprocess_siglip2_end_to_end_with_resize() {
    let p = ImagePreprocessor::siglip2();
    let image = DynTensor::full(&[3, 640, 480], 255.0, crate::DType::F32, &Device::Cpu).unwrap();
    let out = p.preprocess(&image).unwrap();
    assert_eq!(out.dims(), &[3, 384, 384]);
    let data = out.to_flat_vec::<f32>().unwrap();
    for (i, &v) in data.iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-3, "pixel {i}: expected 1.0, got {v}");
    }
}

/// Full ViT-Base end-to-end: [3, 512, 512] -> resize 224x224 -> rescale -> normalize.
#[test]
fn test_preprocess_vit_base_end_to_end_with_resize() {
    let p = ImagePreprocessor::vit_base();
    let image = DynTensor::full(&[3, 512, 512], 128.0, crate::DType::F32, &Device::Cpu).unwrap();
    let out = p.preprocess(&image).unwrap();
    assert_eq!(out.dims(), &[3, 224, 224]);
    let data = out.to_flat_vec::<f32>().unwrap();
    for (i, &v) in data.iter().enumerate() {
        assert!(v.is_finite(), "pixel {i}: non-finite value {v}");
    }
}

// ---------------------------------------------------------------------------
// HWC -> CHW layout conversion (#3531)
// ---------------------------------------------------------------------------

/// [H, W, 3] HWC input is automatically transposed to [3, H, W] CHW.
#[test]
fn test_preprocess_hwc_to_chw_rank3() {
    let p = ImagePreprocessor::new(4, 4, [0.0; 3], [1.0; 3], 1.0).unwrap();
    // Build [4, 4, 3] HWC: R=1.0, G=2.0, B=3.0 everywhere.
    let mut data = Vec::with_capacity(4 * 4 * 3);
    for _ in 0..(4 * 4) {
        data.push(1.0f32); // R
        data.push(2.0); // G
        data.push(3.0); // B
    }
    let image = DynTensor::from_vec(data, &[4, 4, 3], &Device::Cpu).unwrap();
    let out = p.preprocess(&image).unwrap();
    assert_eq!(out.dims(), &[3, 4, 4]);
    let out_data = out.to_flat_vec::<f32>().unwrap();

    // R channel (first 16 values) should all be 1.0.
    for i in 0..16 {
        assert!(
            (out_data[i] - 1.0).abs() < 1e-5,
            "R[{i}]: expected 1.0, got {}",
            out_data[i]
        );
    }
    // G channel (next 16) should all be 2.0.
    for i in 16..32 {
        assert!(
            (out_data[i] - 2.0).abs() < 1e-5,
            "G[{i}]: expected 2.0, got {}",
            out_data[i]
        );
    }
    // B channel (last 16) should all be 3.0.
    for i in 32..48 {
        assert!(
            (out_data[i] - 3.0).abs() < 1e-5,
            "B[{i}]: expected 3.0, got {}",
            out_data[i]
        );
    }
}

/// [B, H, W, 3] BHWC input is transposed to [B, 3, H, W] BCHW.
#[test]
fn test_preprocess_bhwc_to_bchw_rank4() {
    let p = ImagePreprocessor::new(2, 2, [0.0; 3], [1.0; 3], 1.0).unwrap();
    // [2, 2, 2, 3] BHWC with R=1, G=2, B=3.
    let mut data = Vec::with_capacity(2 * 2 * 2 * 3);
    for _ in 0..(2 * 2 * 2) {
        data.push(1.0f32);
        data.push(2.0);
        data.push(3.0);
    }
    let image = DynTensor::from_vec(data, &[2, 2, 2, 3], &Device::Cpu).unwrap();
    let out = p.preprocess(&image).unwrap();
    assert_eq!(out.dims(), &[2, 3, 2, 2]);

    let out_data = out.to_flat_vec::<f32>().unwrap();
    // Batch 0, R channel (indices 0..4) should all be 1.0.
    for i in 0..4 {
        assert!(
            (out_data[i] - 1.0).abs() < 1e-5,
            "B0 R[{i}]: expected 1.0, got {}",
            out_data[i]
        );
    }
}

/// HWC input with resize: [8, 8, 3] -> CHW -> resize to [3, 4, 4].
#[test]
fn test_preprocess_hwc_with_resize() {
    let p = ImagePreprocessor::new(4, 4, [0.0; 3], [1.0; 3], 1.0).unwrap();
    let image = DynTensor::full(&[8, 8, 3], 0.6, crate::DType::F32, &Device::Cpu).unwrap();
    let out = p.preprocess(&image).unwrap();
    assert_eq!(out.dims(), &[3, 4, 4]);
    let data = out.to_flat_vec::<f32>().unwrap();
    for (i, &v) in data.iter().enumerate() {
        assert!((v - 0.6).abs() < 1e-4, "pixel {i}: expected 0.6, got {v}");
    }
}
