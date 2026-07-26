// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for dpdf document inference pipeline models.
//!
//! Measures forward-pass throughput for each dpdf model component using
//! synthetic zero weights on CPU. For large models (Qwen3-VL 2B,
//! GLM-OCR 0.9B), benchmarks a single decoder layer instead of the full
//! model to keep per-iteration time reasonable for CI.
//!
//! Run: `cargo bench -p nn-models --bench dpdf_bench`
//!
//! Part of #3891.

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::vision::{nms_filter, Detection};
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

use nn_models::doclayout_yolo::{DocLayoutYolo, DocLayoutYoloConfig};
use nn_models::dpdf_pipeline::{DpdfPipeline, PipelineConfig};
use nn_models::glm_ocr::{GlmDecoderLayer, GlmOcrConfig};
use nn_models::granite_docling::{GraniteDecoderLayer, GraniteDoclingConfig};
use nn_models::paddle_ocr::PaddleOcrVlConfig;
use nn_models::qwen3_vl::{Qwen3DecoderLayer, Qwen3VLConfig};
use nn_models::table_transformer::{TableTransformer, TableTransformerConfig};

fn cpu() -> Device {
    Device::Cpu
}

// ---------------------------------------------------------------------------
// DocLayout-YOLO: full backbone + neck + head forward
// ---------------------------------------------------------------------------

fn bench_doclayout_yolo_forward(c: &mut Criterion) {
    let cfg = DocLayoutYoloConfig::default();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = DocLayoutYolo::load(&vb, cfg).expect("DocLayoutYolo load failed");

    // Use reduced 320x320 input for fast benchmark iterations.
    let image = DynTensor::zeros(&[1, 3, 320, 320], DType::F32, &cpu())
        .expect("input tensor creation failed");

    c.bench_function("doclayout_yolo_forward_320", |b| {
        b.iter(|| {
            let result = model.forward(black_box(&image));
            black_box(result).unwrap();
        });
    });
}

// ---------------------------------------------------------------------------
// Granite-Docling: single decoder layer (full model too large for fast bench)
// ---------------------------------------------------------------------------

fn bench_granite_docling_forward(c: &mut Criterion) {
    let cfg = GraniteDoclingConfig::default_258m();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let layer_vb = vb.pp("model").pp("layers").pp("0");
    let layer =
        GraniteDecoderLayer::load(&layer_vb, &cfg).expect("GraniteDecoderLayer load failed");

    // [1, 32, 768] — short sequence for decoder layer benchmark.
    let input = DynTensor::zeros(&[1, 32, cfg.decoder_hidden], DType::F32, &cpu())
        .expect("input tensor creation failed");

    c.bench_function("granite_docling_decoder_layer_seq32", |b| {
        b.iter(|| {
            let result = layer.forward(black_box(&input), None);
            black_box(result).unwrap();
        });
    });
}

// ---------------------------------------------------------------------------
// Table Transformer: full DETR forward (ResNet-18 backbone + encoder/decoder)
// ---------------------------------------------------------------------------

fn bench_table_transformer_forward(c: &mut Criterion) {
    let cfg = TableTransformerConfig::preset_detection();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = TableTransformer::load(&vb, &cfg).expect("TableTransformer load failed");

    // Reduced 256x256 input (must be divisible by 32).
    let image = DynTensor::zeros(&[1, 3, 256, 256], DType::F32, &cpu())
        .expect("input tensor creation failed");

    c.bench_function("table_transformer_forward_256", |b| {
        b.iter(|| {
            let result = model.forward(black_box(&image));
            black_box(result).unwrap();
        });
    });
}

// ---------------------------------------------------------------------------
// GLM-OCR: single decoder layer (0.9B model too large for full forward)
// ---------------------------------------------------------------------------

fn bench_glm_ocr_forward(c: &mut Criterion) {
    let cfg = GlmOcrConfig::preset_900m();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let layer_vb = vb.pp("model").pp("layers").pp("0");
    let layer = GlmDecoderLayer::load(&layer_vb, &cfg).expect("GlmDecoderLayer load failed");

    // [1, 32, 1536] — short sequence for decoder layer benchmark.
    let input = DynTensor::zeros(&[1, 32, cfg.hidden_size], DType::F32, &cpu())
        .expect("input tensor creation failed");

    c.bench_function("glm_ocr_decoder_layer_seq32", |b| {
        b.iter(|| {
            let result = layer.forward(black_box(&input), None);
            black_box(result).unwrap();
        });
    });
}

// ---------------------------------------------------------------------------
// Qwen3-VL: single decoder layer (2B model too large for full forward)
// ---------------------------------------------------------------------------

fn bench_qwen3_vl_forward(c: &mut Criterion) {
    let cfg = Qwen3VLConfig::preset_2b();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let layer_vb = vb.pp("model").pp("layers").pp("0");
    let layer = Qwen3DecoderLayer::load(&layer_vb, &cfg).expect("Qwen3DecoderLayer load failed");

    // [1, 32, 1536] — short sequence for decoder layer benchmark.
    let input = DynTensor::zeros(&[1, 32, cfg.hidden_size], DType::F32, &cpu())
        .expect("input tensor creation failed");

    c.bench_function("qwen3_vl_decoder_layer_seq32", |b| {
        b.iter(|| {
            let result = layer.forward(black_box(&input), None);
            black_box(result).unwrap();
        });
    });
}

// ---------------------------------------------------------------------------
// PaddleOCR-VL: vision encode forward
// ---------------------------------------------------------------------------

fn bench_paddle_ocr_vl_vision_encode(c: &mut Criterion) {
    use nn_models::paddle_ocr::PaddleOcrVl;

    let cfg = PaddleOcrVlConfig::default_vl();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = PaddleOcrVl::load(&vb, cfg).expect("PaddleOcrVl load failed");

    // Reduced 224x224 input for fast vision-encode benchmark.
    let image = DynTensor::zeros(&[1, 3, 224, 224], DType::F32, &cpu())
        .expect("input tensor creation failed");

    c.bench_function("paddle_ocr_vl_vision_encode_224", |b| {
        b.iter(|| {
            let result = model.vision_encode(black_box(&image));
            black_box(result).unwrap();
        });
    });
}

// ---------------------------------------------------------------------------
// Pipeline preprocessing: detections_to_regions + build_page
// ---------------------------------------------------------------------------

fn bench_dpdf_pipeline_preprocessing(c: &mut Criterion) {
    let pipeline = DpdfPipeline::new(PipelineConfig::default());

    // Simulate a typical document page with 50 detections.
    let detections: Vec<(usize, f32, [f32; 4])> = (0..50)
        .map(|i| {
            let class_id = i % 10;
            let y_start = (i as f32) * 15.0;
            (
                class_id,
                0.8 + (i as f32) * 0.002,
                [10.0, y_start, 590.0, y_start + 12.0],
            )
        })
        .collect();

    c.bench_function("dpdf_pipeline_preprocessing_50det", |b| {
        b.iter(|| {
            let regions = DpdfPipeline::detections_to_regions(black_box(&detections));
            let page = pipeline.build_page(regions, 612, 792);
            black_box(page);
        });
    });
}

// ---------------------------------------------------------------------------
// NMS postprocessing: filter 500 random detections
// ---------------------------------------------------------------------------

fn bench_nms_postprocessing(c: &mut Criterion) {
    // Generate 500 detections with overlapping boxes across 10 classes.
    let detections: Vec<Detection> = (0..500)
        .map(|i| {
            let class_id = (i % 10) as u32;
            let row = (i / 20) as f32;
            let col = (i % 20) as f32;
            Detection {
                x1: col * 40.0,
                y1: row * 32.0,
                x2: col * 40.0 + 50.0, // overlapping by 10px
                y2: row * 32.0 + 40.0, // overlapping by 8px
                confidence: 0.1 + (i as f32) * 0.0016,
                class_id,
            }
        })
        .collect();

    c.bench_function("nms_filter_500det", |b| {
        b.iter(|| {
            let result = nms_filter(black_box(&detections), 0.25, 0.45);
            black_box(result).unwrap();
        });
    });
}

// ---------------------------------------------------------------------------
// Reading order computation: sort 100 regions
// ---------------------------------------------------------------------------

fn bench_reading_order(c: &mut Criterion) {
    // Build 100 document regions with varied positions and types.
    let detections: Vec<(usize, f32, [f32; 4])> = (0..100)
        .map(|i| {
            let class_id = i % 10;
            // Scatter across the page to exercise the sort.
            let x = ((i * 37) % 50) as f32 * 10.0;
            let y = ((i * 13) % 80) as f32 * 10.0;
            (class_id, 0.9, [x, y, x + 100.0, y + 20.0])
        })
        .collect();
    let regions = DpdfPipeline::detections_to_regions(&detections);

    c.bench_function("reading_order_100regions", |b| {
        b.iter(|| {
            let order = DpdfPipeline::compute_reading_order(black_box(&regions));
            black_box(order);
        });
    });
}

criterion_group!(
    benches,
    bench_doclayout_yolo_forward,
    bench_granite_docling_forward,
    bench_table_transformer_forward,
    bench_glm_ocr_forward,
    bench_qwen3_vl_forward,
    bench_paddle_ocr_vl_vision_encode,
    bench_dpdf_pipeline_preprocessing,
    bench_nms_postprocessing,
    bench_reading_order,
);
criterion_main!(benches);
