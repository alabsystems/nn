// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for DpdfPipelineMetal GPU dispatch.
//!
//! Tests verify image preprocessing, model dispatch, and pipeline integration
//! on Metal GPU. The model forward path runs through the registered Metal
//! DynTensor backend; NMS and reading order remain on CPU.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_models::doclayout_yolo::INPUT_SIZE;
use nn_models::dpdf_pipeline::PipelineConfig;

use super::{extract_image_hw, DpdfPipelineMetal, IMAGENET_MEAN, IMAGENET_STD};
use crate::test_common::init;

/// Create random f32 data using the test PRNG.
fn rand_f32_vec(seed: u64, count: usize, lo: f32, hi: f32) -> Vec<f32> {
    nn_core::test_prng::rand_f32_vec(seed, count, lo, hi)
}

/// Create a random image tensor in HWC format `[H, W, 3]` on the given device.
fn random_image_hwc(seed: u64, h: usize, w: usize, device: &Device) -> DynTensor {
    let data = rand_f32_vec(seed, h * w * 3, 0.0, 1.0);
    DynTensor::from_vec(data, &[h, w, 3], device).unwrap()
}

/// Create a random image tensor in CHW format `[3, H, W]` on the given device.
fn random_image_chw(seed: u64, h: usize, w: usize, device: &Device) -> DynTensor {
    let data = rand_f32_vec(seed, 3 * h * w, 0.0, 1.0);
    DynTensor::from_vec(data, &[3, h, w], device).unwrap()
}

/// Create a random batched image tensor `[1, 3, H, W]` on the given device.
fn random_image_bchw(seed: u64, h: usize, w: usize, device: &Device) -> DynTensor {
    let data = rand_f32_vec(seed, 3 * h * w, 0.0, 1.0);
    DynTensor::from_vec(data, &[1, 3, h, w], device).unwrap()
}

// ---------------------------------------------------------------------------
// extract_image_hw tests
// ---------------------------------------------------------------------------

#[test]
fn test_extract_hw_hwc() {
    let img = DynTensor::from_vec(vec![0.0; 480 * 640 * 3], &[480, 640, 3], &Device::Cpu).unwrap();
    let (h, w) = extract_image_hw(&img).unwrap();
    assert_eq!((h, w), (480, 640));
}

#[test]
fn test_extract_hw_chw() {
    let img = DynTensor::from_vec(vec![0.0; 3 * 480 * 640], &[3, 480, 640], &Device::Cpu).unwrap();
    let (h, w) = extract_image_hw(&img).unwrap();
    assert_eq!((h, w), (480, 640));
}

#[test]
fn test_extract_hw_bchw() {
    let img =
        DynTensor::from_vec(vec![0.0; 3 * 480 * 640], &[1, 3, 480, 640], &Device::Cpu).unwrap();
    let (h, w) = extract_image_hw(&img).unwrap();
    assert_eq!((h, w), (480, 640));
}

#[test]
fn test_extract_hw_invalid_rank() {
    let img = DynTensor::from_vec(vec![0.0; 12], &[3, 4], &Device::Cpu).unwrap();
    assert!(extract_image_hw(&img).is_err());
}

#[test]
fn test_extract_hw_invalid_channels() {
    let img = DynTensor::from_vec(vec![0.0; 4 * 8 * 5], &[4, 8, 5], &Device::Cpu).unwrap();
    assert!(extract_image_hw(&img).is_err());
}

// ---------------------------------------------------------------------------
// preprocess_image tests
// ---------------------------------------------------------------------------

#[test]
fn test_preprocess_hwc_shape() {
    init();
    let device = Device::metal();
    let img = random_image_hwc(42, 480, 640, &device);

    let pipeline = create_test_pipeline_no_weights();

    let result = pipeline.preprocess_image(&img).unwrap();
    assert_eq!(result.dims(), &[1, 3, INPUT_SIZE, INPUT_SIZE]);
    assert!(result.device().is_gpu(), "preprocessed tensor should be on GPU");
}

#[test]
fn test_preprocess_chw_shape() {
    init();
    let device = Device::metal();
    let img = random_image_chw(43, 600, 400, &device);

    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();
    assert_eq!(result.dims(), &[1, 3, INPUT_SIZE, INPUT_SIZE]);
    assert!(result.device().is_gpu());
}

#[test]
fn test_preprocess_bchw_shape() {
    init();
    let device = Device::metal();
    let img = random_image_bchw(44, 320, 240, &device);

    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();
    assert_eq!(result.dims(), &[1, 3, INPUT_SIZE, INPUT_SIZE]);
    assert!(result.device().is_gpu());
}

#[test]
fn test_preprocess_cpu_input_uploads() {
    init();
    let img = random_image_hwc(45, 100, 100, &Device::Cpu);

    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();
    assert_eq!(result.dims(), &[1, 3, INPUT_SIZE, INPUT_SIZE]);
    assert!(result.device().is_gpu(), "CPU input should be uploaded to GPU");
}

#[test]
fn test_preprocess_normalization_range() {
    init();
    let device = Device::metal();
    // Create a uniform 0.5 image -- after normalization with ImageNet stats,
    // each channel should be (0.5 - mean) / std.
    let data = vec![0.5_f32; 3 * 64 * 64];
    let img = DynTensor::from_vec(data, &[3, 64, 64], &device).unwrap();

    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();

    let cpu_result = result.to_device(&Device::Cpu).unwrap();
    let vals = cpu_result.to_flat_vec::<f32>().unwrap();

    // Expected per-channel values after normalization.
    let expected_r = (0.5 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    let expected_g = (0.5 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    let expected_b = (0.5 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];

    let pixels_per_channel = INPUT_SIZE * INPUT_SIZE;

    // Check a few pixels from each channel.
    let tol = 0.05; // bilinear resize may introduce slight variation
    let r_val = vals[0]; // first pixel of R channel
    let g_val = vals[pixels_per_channel]; // first pixel of G channel
    let b_val = vals[2 * pixels_per_channel]; // first pixel of B channel

    assert!(
        (r_val - expected_r).abs() < tol,
        "R channel: got {r_val}, expected ~{expected_r}"
    );
    assert!(
        (g_val - expected_g).abs() < tol,
        "G channel: got {g_val}, expected ~{expected_g}"
    );
    assert!(
        (b_val - expected_b).abs() < tol,
        "B channel: got {b_val}, expected ~{expected_b}"
    );
}

#[test]
fn test_preprocess_square_image() {
    init();
    let device = Device::metal();
    // Square image at model input size should be identity-sized resize.
    let img = random_image_chw(46, INPUT_SIZE, INPUT_SIZE, &device);

    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();
    assert_eq!(result.dims(), &[1, 3, INPUT_SIZE, INPUT_SIZE]);
    assert!(result.device().is_gpu());
}

#[test]
fn test_preprocess_invalid_rank() {
    init();
    let device = Device::metal();
    let img = DynTensor::from_vec(vec![0.0; 12], &[3, 4], &device).unwrap();

    let pipeline = create_test_pipeline_no_weights();
    assert!(pipeline.preprocess_image(&img).is_err());
}

#[test]
fn test_preprocess_invalid_batch() {
    init();
    let device = Device::metal();
    // batch=2 is not supported
    let img = DynTensor::from_vec(
        vec![0.0; 2 * 3 * 64 * 64],
        &[2, 3, 64, 64],
        &device,
    )
    .unwrap();

    let pipeline = create_test_pipeline_no_weights();
    assert!(pipeline.preprocess_image(&img).is_err());
}

#[test]
fn test_preprocess_invalid_channels_4d() {
    init();
    let device = Device::metal();
    // 4 channels instead of 3
    let img = DynTensor::from_vec(
        vec![0.0; 4 * 32 * 32],
        &[1, 4, 32, 32],
        &device,
    )
    .unwrap();

    let pipeline = create_test_pipeline_no_weights();
    assert!(pipeline.preprocess_image(&img).is_err());
}

// ---------------------------------------------------------------------------
// Pipeline integration (detection post-processing)
// ---------------------------------------------------------------------------

#[test]
fn test_pipeline_config_accessors() {
    init();
    let config = PipelineConfig {
        layout_conf_threshold: 0.3,
        layout_iou_threshold: 0.5,
        ocr_max_tokens: 512,
        enable_table_structure: false,
        postprocess_config: nn_models::dpdf_postprocess::PostProcessConfig::default(),
        table_structure_config: nn_models::table_structure::TableStructureConfig::default(),
    };
    let pipeline = create_test_pipeline_no_weights_with_config(config);

    assert!((pipeline.config().layout_conf_threshold - 0.3).abs() < 1e-6);
    assert!((pipeline.config().layout_iou_threshold - 0.5).abs() < 1e-6);
    assert_eq!(pipeline.config().ocr_max_tokens, 512);
    assert!(!pipeline.config().enable_table_structure);
}

#[test]
fn test_process_document_empty() {
    init();
    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.process_document(&[]);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().pages.len(), 0);
}

#[test]
fn test_model_accessor() {
    init();
    let pipeline = create_test_pipeline_no_weights();
    // Verify model accessor returns the configured model.
    assert_eq!(pipeline.model().config().num_classes, 10);
    assert_eq!(pipeline.model().config().input_channels, 3);
}

// ---------------------------------------------------------------------------
// GPU image preprocessing dispatch per model preset
// ---------------------------------------------------------------------------

/// Helper: run the DpdfImagePreprocessMetal pipeline for a preset and verify
/// the output shape, device, dtype, and finiteness.
fn assert_preprocess_preset(
    config: nn_models::dpdf_image_preprocess::DpdfPreprocessConfig,
    src_h: usize,
    src_w: usize,
    expected_h: usize,
    expected_w: usize,
    label: &str,
) {
    init();
    let device = Device::metal();
    let img = random_image_hwc(100, src_h, src_w, &device);

    let pp = crate::dpdf_image_preprocess_metal::DpdfImagePreprocessMetal::new(config);
    let result = pp.preprocess_image(&img).unwrap();

    // Shape check: output must be CHW with expected spatial dimensions.
    let dims = result.dims();
    assert_eq!(
        dims[0], 3,
        "{label}: expected 3 channels, got {}",
        dims[0]
    );
    assert_eq!(
        dims[1], expected_h,
        "{label}: expected height {expected_h}, got {}",
        dims[1]
    );
    assert_eq!(
        dims[2], expected_w,
        "{label}: expected width {expected_w}, got {}",
        dims[2]
    );

    // Device: must remain on GPU.
    assert!(result.device().is_gpu(), "{label}: result must be on GPU");

    // Dtype: must be F32.
    assert_eq!(result.dtype(), DType::F32, "{label}: dtype must be F32");

    // Finiteness: no NaN or Inf.
    let cpu = result.to_device(&Device::Cpu).unwrap();
    let vals = cpu.to_flat_vec::<f32>().unwrap();
    for (i, v) in vals.iter().enumerate() {
        assert!(
            v.is_finite(),
            "{label}: non-finite value at index {i}: {v}"
        );
    }
}

#[test]
fn test_preset_granite_docling_dispatch() {
    let config = nn_models::dpdf_image_preprocess::DpdfPreprocessConfig::for_granite_docling();
    // Granite Docling: 384x384 fixed, no aspect ratio.
    assert_preprocess_preset(config, 480, 640, 384, 384, "granite_docling");
}

#[test]
fn test_preset_doclayout_yolo_dispatch() {
    let config = nn_models::dpdf_image_preprocess::DpdfPreprocessConfig::for_doclayout_yolo();
    // DocLayout YOLO: 1024x1024 letterbox, maintains aspect.
    assert_preprocess_preset(config, 768, 512, 1024, 1024, "doclayout_yolo");
}

#[test]
fn test_preset_paddle_ocr_detect_dispatch() {
    let config = nn_models::dpdf_image_preprocess::DpdfPreprocessConfig::for_paddle_ocr_detect();
    // PaddleOCR detect: 960 max side, aspect-preserving.
    // 480x960 -> scale = min(960/480, 960/960) = 1.0 -> 480x960.
    assert_preprocess_preset(config, 480, 960, 480, 960, "paddle_ocr_detect");
}

#[test]
fn test_preset_paddle_ocr_recognize_dispatch() {
    let config =
        nn_models::dpdf_image_preprocess::DpdfPreprocessConfig::for_paddle_ocr_recognize();
    // PaddleOCR recognize: 48x320, aspect-preserving.
    // 96x640 -> scale = min(48/96, 320/640) = 0.5 -> 48x320.
    assert_preprocess_preset(config, 96, 640, 48, 320, "paddle_ocr_recognize");
}

#[test]
fn test_preset_table_transformer_dispatch() {
    let config =
        nn_models::dpdf_image_preprocess::DpdfPreprocessConfig::for_table_transformer();
    // Table Transformer: 800x800, aspect-preserving.
    // 600x800 -> scale = min(800/600, 800/800) = 1.0 -> 600x800.
    assert_preprocess_preset(config, 600, 800, 600, 800, "table_transformer");
}

#[test]
fn test_preset_qwen3_vl_dispatch() {
    init();
    let device = Device::metal();
    let config = nn_models::dpdf_image_preprocess::DpdfPreprocessConfig::for_qwen3_vl();
    // Qwen3-VL: dynamic resolution (target_height/width=0, maintain_aspect=true).
    // compute_resize_dims with target 0 returns (1, 1) -- minimal.
    let img = random_image_hwc(101, 224, 224, &device);
    let pp = crate::dpdf_image_preprocess_metal::DpdfImagePreprocessMetal::new(config);
    let result = pp.preprocess_image(&img).unwrap();

    // Verify CHW output and finiteness (dynamic dims).
    assert_eq!(result.dims()[0], 3, "qwen3_vl: must have 3 channels");
    assert!(result.device().is_gpu(), "qwen3_vl: result must be on GPU");
    let cpu = result.to_device(&Device::Cpu).unwrap();
    let vals = cpu.to_flat_vec::<f32>().unwrap();
    for (i, v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "qwen3_vl: non-finite at {i}: {v}");
    }
}

#[test]
fn test_preset_glm_ocr_dispatch() {
    let config = nn_models::dpdf_image_preprocess::DpdfPreprocessConfig::for_glm_ocr();
    // GLM-OCR: resize longest side to 1120, aspect-preserving, PaddingMode::None.
    // 800x600 -> scale = min(1120/800, 1120/600) = 1.4 -> 1120x840 (no padding).
    assert_preprocess_preset(config, 800, 600, 1120, 840, "glm_ocr");
}

// ---------------------------------------------------------------------------
// GPU buffer allocation and lifecycle
// ---------------------------------------------------------------------------

#[test]
fn test_gpu_buffer_allocation_hwc() {
    init();
    let device = Device::metal();
    let img = random_image_hwc(200, 128, 128, &device);

    // Verify input is on GPU.
    assert!(img.device().is_gpu(), "input must be allocated on GPU");

    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();

    // Verify output is on GPU and has correct shape.
    assert!(result.device().is_gpu(), "output must remain on GPU");
    assert_eq!(result.dims(), &[1, 3, INPUT_SIZE, INPUT_SIZE]);

    // Read back to CPU to verify buffer is valid and readable.
    let cpu = result.to_device(&Device::Cpu).unwrap();
    let vals = cpu.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        vals.len(),
        1 * 3 * INPUT_SIZE * INPUT_SIZE,
        "flat vec length must match shape"
    );
}

#[test]
fn test_gpu_buffer_lifecycle_multiple_tensors() {
    init();
    let device = Device::metal();

    // Allocate several tensors on GPU, process them, and verify all are valid.
    let imgs: Vec<DynTensor> = (0..5)
        .map(|i| random_image_chw(300 + i, 64, 64, &device))
        .collect();

    let pipeline = create_test_pipeline_no_weights();
    for (i, img) in imgs.iter().enumerate() {
        let result = pipeline.preprocess_image(img).unwrap();
        assert!(
            result.device().is_gpu(),
            "tensor {i}: output must be on GPU"
        );
        assert_eq!(
            result.dims(),
            &[1, 3, INPUT_SIZE, INPUT_SIZE],
            "tensor {i}: incorrect shape"
        );
    }
}

#[test]
fn test_gpu_buffer_cpu_upload_roundtrip() {
    init();
    // Start on CPU, preprocess uploads to GPU, read back to CPU.
    let img = random_image_hwc(201, 100, 100, &Device::Cpu);
    assert!(!img.device().is_gpu(), "input should start on CPU");

    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();
    assert!(result.device().is_gpu(), "output should be on GPU after preprocess");

    let cpu_result = result.to_device(&Device::Cpu).unwrap();
    assert!(!cpu_result.device().is_gpu(), "after to_device(Cpu) should be on CPU");
    assert_eq!(cpu_result.dims(), &[1, 3, INPUT_SIZE, INPUT_SIZE]);
}

// ---------------------------------------------------------------------------
// Batch processing (multiple images in sequence)
// ---------------------------------------------------------------------------

#[test]
fn test_batch_processing_sequential_pages() {
    init();
    let device = Device::metal();
    let page_count = 4;

    let pages: Vec<DynTensor> = (0..page_count)
        .map(|i| random_image_hwc(400 + i, 480, 640, &device))
        .collect();

    let pipeline = create_test_pipeline_no_weights();

    // Process each page and verify shape/device consistency.
    let mut results = Vec::new();
    for (i, page) in pages.iter().enumerate() {
        let preprocessed = pipeline.preprocess_image(page).unwrap();
        assert_eq!(
            preprocessed.dims(),
            &[1, 3, INPUT_SIZE, INPUT_SIZE],
            "page {i}: incorrect shape"
        );
        assert!(
            preprocessed.device().is_gpu(),
            "page {i}: output must be on GPU"
        );
        results.push(preprocessed);
    }

    assert_eq!(results.len(), page_count as usize);
}

#[test]
fn test_batch_processing_mixed_input_sizes() {
    init();
    let device = Device::metal();

    // Different input sizes should all produce the same output shape.
    let sizes = [(100, 100), (480, 640), (1024, 768), (200, 300)];
    let pipeline = create_test_pipeline_no_weights();

    for (i, (h, w)) in sizes.iter().enumerate() {
        let img = random_image_hwc(500 + i as u64, *h, *w, &device);
        let result = pipeline.preprocess_image(&img).unwrap();
        assert_eq!(
            result.dims(),
            &[1, 3, INPUT_SIZE, INPUT_SIZE],
            "size ({h}x{w}): output shape mismatch"
        );
    }
}

#[test]
fn test_batch_processing_mixed_formats() {
    init();
    let device = Device::metal();
    let pipeline = create_test_pipeline_no_weights();

    // HWC input
    let hwc = random_image_hwc(600, 128, 128, &device);
    let r1 = pipeline.preprocess_image(&hwc).unwrap();
    assert_eq!(r1.dims(), &[1, 3, INPUT_SIZE, INPUT_SIZE]);

    // CHW input
    let chw = random_image_chw(601, 128, 128, &device);
    let r2 = pipeline.preprocess_image(&chw).unwrap();
    assert_eq!(r2.dims(), &[1, 3, INPUT_SIZE, INPUT_SIZE]);

    // BCHW input
    let bchw = random_image_bchw(602, 128, 128, &device);
    let r3 = pipeline.preprocess_image(&bchw).unwrap();
    assert_eq!(r3.dims(), &[1, 3, INPUT_SIZE, INPUT_SIZE]);
}

#[test]
fn test_process_document_empty_pages() {
    init();
    let pipeline = create_test_pipeline_no_weights();
    let doc = pipeline.process_document(&[]).unwrap();
    assert_eq!(doc.pages.len(), 0);
}

// ---------------------------------------------------------------------------
// Preprocessing output shapes match model input requirements
// ---------------------------------------------------------------------------

#[test]
fn test_output_shape_small_image() {
    init();
    let device = Device::metal();
    // Very small image (16x16) should still resize to model input size.
    let img = random_image_hwc(700, 16, 16, &device);
    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();
    assert_eq!(result.dims(), &[1, 3, INPUT_SIZE, INPUT_SIZE]);
}

#[test]
fn test_output_shape_wide_image() {
    init();
    let device = Device::metal();
    // Very wide (panoramic) image.
    let img = random_image_hwc(701, 100, 800, &device);
    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();
    assert_eq!(result.dims(), &[1, 3, INPUT_SIZE, INPUT_SIZE]);
}

#[test]
fn test_output_shape_tall_image() {
    init();
    let device = Device::metal();
    // Very tall image.
    let img = random_image_hwc(702, 800, 100, &device);
    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();
    assert_eq!(result.dims(), &[1, 3, INPUT_SIZE, INPUT_SIZE]);
}

#[test]
fn test_output_shape_exact_model_size() {
    init();
    let device = Device::metal();
    // Input already at model input size should be identity resize.
    let img = random_image_chw(703, INPUT_SIZE, INPUT_SIZE, &device);
    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();
    assert_eq!(result.dims(), &[1, 3, INPUT_SIZE, INPUT_SIZE]);
}

#[test]
fn test_output_dtype_is_f32() {
    init();
    let device = Device::metal();
    let img = random_image_hwc(704, 128, 128, &device);
    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();
    assert_eq!(result.dtype(), DType::F32, "output must be F32");
}

// ---------------------------------------------------------------------------
// Normalization values: ImageNet mean/std produce correct range on GPU
// ---------------------------------------------------------------------------

#[test]
fn test_normalization_all_zeros_image() {
    init();
    let device = Device::metal();
    // All-zero image: (0 - mean) / std per channel.
    let data = vec![0.0_f32; 3 * 64 * 64];
    let img = DynTensor::from_vec(data, &[3, 64, 64], &device).unwrap();

    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();

    let cpu = result.to_device(&Device::Cpu).unwrap();
    let vals = cpu.to_flat_vec::<f32>().unwrap();

    // Expected: (0 - mean) / std
    let expected_r = (0.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    let expected_g = (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    let expected_b = (0.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];

    let pixels_per_channel = INPUT_SIZE * INPUT_SIZE;
    let tol = 0.05; // bilinear interpolation may shift values slightly

    let r_val = vals[0];
    let g_val = vals[pixels_per_channel];
    let b_val = vals[2 * pixels_per_channel];

    assert!(
        (r_val - expected_r).abs() < tol,
        "zero image R: got {r_val}, expected ~{expected_r}"
    );
    assert!(
        (g_val - expected_g).abs() < tol,
        "zero image G: got {g_val}, expected ~{expected_g}"
    );
    assert!(
        (b_val - expected_b).abs() < tol,
        "zero image B: got {b_val}, expected ~{expected_b}"
    );
}

#[test]
fn test_normalization_all_ones_image() {
    init();
    let device = Device::metal();
    // All-one image: (1 - mean) / std per channel.
    let data = vec![1.0_f32; 3 * 64 * 64];
    let img = DynTensor::from_vec(data, &[3, 64, 64], &device).unwrap();

    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();

    let cpu = result.to_device(&Device::Cpu).unwrap();
    let vals = cpu.to_flat_vec::<f32>().unwrap();

    let expected_r = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    let expected_g = (1.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    let expected_b = (1.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];

    let pixels_per_channel = INPUT_SIZE * INPUT_SIZE;
    let tol = 0.05;

    let r_val = vals[0];
    let g_val = vals[pixels_per_channel];
    let b_val = vals[2 * pixels_per_channel];

    assert!(
        (r_val - expected_r).abs() < tol,
        "ones image R: got {r_val}, expected ~{expected_r}"
    );
    assert!(
        (g_val - expected_g).abs() < tol,
        "ones image G: got {g_val}, expected ~{expected_g}"
    );
    assert!(
        (b_val - expected_b).abs() < tol,
        "ones image B: got {b_val}, expected ~{expected_b}"
    );
}

#[test]
fn test_normalization_mid_gray_image() {
    init();
    let device = Device::metal();
    // Mid-gray: 0.5 per channel. After normalization, should be close to:
    // R: (0.5 - 0.485) / 0.229 ~ 0.0655
    // G: (0.5 - 0.456) / 0.224 ~ 0.1964
    // B: (0.5 - 0.406) / 0.225 ~ 0.4178
    let data = vec![0.5_f32; 3 * 64 * 64];
    let img = DynTensor::from_vec(data, &[3, 64, 64], &device).unwrap();

    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();

    let cpu = result.to_device(&Device::Cpu).unwrap();
    let vals = cpu.to_flat_vec::<f32>().unwrap();

    let expected_r = (0.5 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    let expected_g = (0.5 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    let expected_b = (0.5 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];

    let pixels_per_channel = INPUT_SIZE * INPUT_SIZE;
    let tol = 0.05;

    // Check several pixel positions in each channel.
    for offset in [0, 1, pixels_per_channel / 2, pixels_per_channel - 1] {
        let r = vals[offset];
        assert!(
            (r - expected_r).abs() < tol,
            "mid-gray R[{offset}]: got {r}, expected ~{expected_r}"
        );
        let g = vals[pixels_per_channel + offset];
        assert!(
            (g - expected_g).abs() < tol,
            "mid-gray G[{offset}]: got {g}, expected ~{expected_g}"
        );
        let b = vals[2 * pixels_per_channel + offset];
        assert!(
            (b - expected_b).abs() < tol,
            "mid-gray B[{offset}]: got {b}, expected ~{expected_b}"
        );
    }
}

#[test]
fn test_normalization_range_bounded() {
    init();
    let device = Device::metal();
    // Random image in [0, 1]: after ImageNet normalization, values should be
    // within a reasonable range. Min: (0 - max_mean) / min_std ~ -2.12
    // Max: (1 - min_mean) / min_std ~ 2.64
    let img = random_image_chw(750, 128, 128, &device);

    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();

    let cpu = result.to_device(&Device::Cpu).unwrap();
    let vals = cpu.to_flat_vec::<f32>().unwrap();

    let min_val = *vals
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_val = *vals
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    // Theoretical bounds for ImageNet normalization of [0,1] input:
    // min ~ (0 - 0.485) / 0.225 ~ -2.16
    // max ~ (1 - 0.406) / 0.224 ~ 2.65
    assert!(
        min_val > -3.0,
        "min value {min_val} exceeds expected lower bound"
    );
    assert!(
        max_val < 4.0,
        "max value {max_val} exceeds expected upper bound"
    );
}

// ---------------------------------------------------------------------------
// No NaN/Inf in GPU outputs
// ---------------------------------------------------------------------------

#[test]
fn test_no_nan_inf_random_hwc() {
    init();
    let device = Device::metal();
    let img = random_image_hwc(800, 256, 256, &device);

    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();

    let cpu = result.to_device(&Device::Cpu).unwrap();
    let vals = cpu.to_flat_vec::<f32>().unwrap();

    for (i, v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "NaN/Inf at index {i}: {v} (random HWC input)");
    }
}

#[test]
fn test_no_nan_inf_random_chw() {
    init();
    let device = Device::metal();
    let img = random_image_chw(801, 256, 256, &device);

    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();

    let cpu = result.to_device(&Device::Cpu).unwrap();
    let vals = cpu.to_flat_vec::<f32>().unwrap();

    for (i, v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "NaN/Inf at index {i}: {v} (random CHW input)");
    }
}

#[test]
fn test_no_nan_inf_random_bchw() {
    init();
    let device = Device::metal();
    let img = random_image_bchw(802, 256, 256, &device);

    let pipeline = create_test_pipeline_no_weights();
    let result = pipeline.preprocess_image(&img).unwrap();

    let cpu = result.to_device(&Device::Cpu).unwrap();
    let vals = cpu.to_flat_vec::<f32>().unwrap();

    for (i, v) in vals.iter().enumerate() {
        assert!(
            v.is_finite(),
            "NaN/Inf at index {i}: {v} (random BCHW input)"
        );
    }
}

#[test]
fn test_no_nan_inf_extreme_values() {
    init();
    let device = Device::metal();
    // Edge case: pixel values at exact boundaries.
    let data = vec![0.0_f32; 3 * 32 * 32];
    let zero_img = DynTensor::from_vec(data, &[3, 32, 32], &device).unwrap();
    let one_data = vec![1.0_f32; 3 * 32 * 32];
    let one_img = DynTensor::from_vec(one_data, &[3, 32, 32], &device).unwrap();

    let pipeline = create_test_pipeline_no_weights();

    for (img, label) in [(&zero_img, "all-zero"), (&one_img, "all-one")] {
        let result = pipeline.preprocess_image(img).unwrap();
        let cpu = result.to_device(&Device::Cpu).unwrap();
        let vals = cpu.to_flat_vec::<f32>().unwrap();
        for (i, v) in vals.iter().enumerate() {
            assert!(
                v.is_finite(),
                "NaN/Inf at index {i}: {v} ({label} image)"
            );
        }
    }
}

#[test]
fn test_no_nan_inf_preprocess_per_preset() {
    init();
    let device = Device::metal();
    let img = random_image_hwc(803, 200, 300, &device);

    let presets: Vec<(
        nn_models::dpdf_image_preprocess::DpdfPreprocessConfig,
        &str,
    )> = vec![
        (
            nn_models::dpdf_image_preprocess::DpdfPreprocessConfig::for_granite_docling(),
            "granite_docling",
        ),
        (
            nn_models::dpdf_image_preprocess::DpdfPreprocessConfig::for_doclayout_yolo(),
            "doclayout_yolo",
        ),
        (
            nn_models::dpdf_image_preprocess::DpdfPreprocessConfig::for_paddle_ocr_detect(),
            "paddle_ocr_detect",
        ),
        (
            nn_models::dpdf_image_preprocess::DpdfPreprocessConfig::for_paddle_ocr_recognize(),
            "paddle_ocr_recognize",
        ),
        (
            nn_models::dpdf_image_preprocess::DpdfPreprocessConfig::for_table_transformer(),
            "table_transformer",
        ),
        (
            nn_models::dpdf_image_preprocess::DpdfPreprocessConfig::for_qwen3_vl(),
            "qwen3_vl",
        ),
        (
            nn_models::dpdf_image_preprocess::DpdfPreprocessConfig::for_glm_ocr(),
            "glm_ocr",
        ),
    ];

    for (config, label) in presets {
        let pp = crate::dpdf_image_preprocess_metal::DpdfImagePreprocessMetal::new(config);
        let result = pp.preprocess_image(&img).unwrap();
        let cpu = result.to_device(&Device::Cpu).unwrap();
        let vals = cpu.to_flat_vec::<f32>().unwrap();
        for (i, v) in vals.iter().enumerate() {
            assert!(
                v.is_finite(),
                "{label}: NaN/Inf at index {i}: {v}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a test pipeline without model weights.
///
/// Uses the DocLayoutYolo model with zeroed weights (ZerosBackend).
/// Suitable for testing preprocessing and pipeline configuration;
/// model forward output will be numerically meaningless but structurally valid.
fn create_test_pipeline_no_weights() -> DpdfPipelineMetal {
    create_test_pipeline_no_weights_with_config(PipelineConfig::default())
}

/// Create a test pipeline with specified config and no real weights.
fn create_test_pipeline_no_weights_with_config(config: PipelineConfig) -> DpdfPipelineMetal {
    let device = Device::metal();
    let yolo_config = nn_models::doclayout_yolo::DocLayoutYoloConfig::default();
    let vb = nn_core::var_builder::VarBuilder::zeros(DType::F32, &device);
    let model = nn_models::doclayout_yolo::DocLayoutYolo::load(&vb, yolo_config)
        .expect("DocLayoutYolo::load with zeros VarBuilder");
    DpdfPipelineMetal::new(model, config)
}
