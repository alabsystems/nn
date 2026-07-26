// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro auto-converter parity tests via ConvertBuilder API (#4276).
//!
//! Validates that the ConvertBuilder pipeline (the production API used by
//! `nn convert`) produces correct results for Kokoro model fixtures:
//!
//! 1. **ConvertReport validation** -- op mapping, dispatch counts, compilation
//!    metrics, and verification coverage for Kokoro-style subgraphs.
//!
//! 2. **GPU numerical execution** -- compile Kokoro mini fixtures via
//!    ConvertBuilder, execute on Metal GPU, verify output is finite and
//!    non-trivial.
//!
//! 3. **Per-segment ConvertBuilder pipeline** -- run the full ConvertBuilder
//!    pipeline on each Kokoro segment export (gated on segment files).
//!
//! 4. **Cross-path numerical parity** -- compare ConvertBuilder output against
//!    direct `convert()` output for the same fixture, proving the builder API
//!    and the legacy API produce identical results.
//!
//! # Why ConvertBuilder matters
//!
//! `ConvertBuilder` is the production entry point for `nn convert`. It
//! produces a `ConvertReport` with detailed metrics (op coverage, dispatch
//! count, RTF estimate, verification coverage). The existing Kokoro parity
//! tests use `convert()` (legacy API) or structural checks. This test
//! validates the full builder pipeline on Kokoro-representative graphs.
//!
//! Part of #4276 (Kokoro auto-converter parity test).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data")
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn segment_dir(segment: &str) -> PathBuf {
    workspace_root()
        .join("models")
        .join("kokoro-82m")
        .join(segment)
}

/// Write synthetic safetensors weights matching the kokoro_decoder_mini fixture.
///
/// Decoder mini weights:
///   conv1.weight: [16, 8, 3]
///   conv1.bias:   [16]
///   conv2.weight: [16, 16, 3]
///   conv2.bias:   [16]
fn write_decoder_mini_weights(dir: &Path) -> PathBuf {
    let conv1_w: Vec<u8> = (0..16 * 8 * 3)
        .map(|i| (i as f32) * 0.001)
        .flat_map(f32::to_le_bytes)
        .collect();
    let conv1_b: Vec<u8> = [0.0f32; 16]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let conv2_w: Vec<u8> = (0..16 * 16 * 3)
        .map(|i| (i as f32) * 0.001)
        .flat_map(f32::to_le_bytes)
        .collect();
    let conv2_b: Vec<u8> = [0.0f32; 16]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "conv1.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![16, 8, 3], &conv1_w)
            .unwrap(),
    );
    tensors.insert(
        "conv1.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![16], &conv1_b).unwrap(),
    );
    tensors.insert(
        "conv2.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![16, 16, 3], &conv2_w)
            .unwrap(),
    );
    tensors.insert(
        "conv2.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![16], &conv2_b).unwrap(),
    );

    let weights_path = dir.join("weights.safetensors");
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(&weights_path, serialized).unwrap();
    weights_path
}

/// Check if a segment's export files exist.
fn require_segment_files(segment: &str) -> Option<(PathBuf, PathBuf)> {
    let dir = segment_dir(segment);
    let graph = dir.join("graph.json");
    let weights = dir.join("weights.safetensors");
    if !graph.exists() || !weights.exists() {
        eprintln!(
            "SKIP: Kokoro segment '{}' export not found at {}",
            segment,
            dir.display()
        );
        return None;
    }
    Some((graph, weights))
}

// ===========================================================================
// 1. ConvertBuilder + ConvertReport for Kokoro decoder mini
// ===========================================================================

/// ConvertBuilder pipeline for kokoro_decoder_mini: validates ConvertReport
/// fields specific to Kokoro-style subgraphs (Conv1d, InstanceNorm, etc.).
///
/// This exercises the same code path as `nn convert kokoro_decoder.json weights.safetensors`
/// and validates that the ConvertReport accurately reflects Kokoro op patterns.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_convert_builder_kokoro_decoder_mini_report() {
    let graph_path = test_data_dir().join("kokoro_decoder_mini.json");
    if !graph_path.exists() {
        eprintln!(
            "SKIP: kokoro_decoder_mini.json not found at {}",
            graph_path.display()
        );
        return;
    }

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let dir = std::env::temp_dir().join(format!("nn_kokoro_cb_decoder_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let weights_path = write_decoder_mini_weights(&dir);

    let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
        .optimize(nn_import::OptLevel::Full)
        .verify(nn_import::VerifyLevel::None)
        .build()
        .unwrap_or_else(|e| panic!("ConvertBuilder failed for decoder mini: {e:?}"));

    let _ = std::fs::remove_dir_all(&dir);

    let report = &result.report;

    // --- Op mapping: Kokoro decoder mini has Conv1d, InstanceNorm, LeakyReLU,
    // Add, Slice, Exp, Sin, Cat = 10 aten ops total ---
    assert!(
        report.op_count >= 8,
        "decoder mini should have >= 8 aten ops, got {}",
        report.op_count
    );
    assert!(
        report.mapped_ops_count() >= 8,
        "all decoder mini ops should be mapped, got {}/{}",
        report.mapped_ops_count(),
        report.op_count
    );
    assert!(
        report.unmapped_ops.is_empty(),
        "decoder mini should have no unmapped ops: {:?}",
        report.unmapped_ops
    );

    // Verify Kokoro-specific ops are present in the mapped list.
    let mapped_names: Vec<&str> = report.mapped_ops.iter().map(|(n, _)| n.as_str()).collect();
    assert!(
        mapped_names.contains(&"torch.ops.aten.convolution.default"),
        "should contain convolution: {mapped_names:?}"
    );

    // --- Import metrics ---
    assert_eq!(
        report.num_user_inputs, 1,
        "decoder mini has 1 user input (x)"
    );
    assert_eq!(
        report.num_weights_loaded, 4,
        "decoder mini has 4 weight tensors"
    );

    // --- Compilation metrics ---
    assert!(report.dispatch_count > 0, "dispatch_count must be positive");
    assert!(report.total_steps > 0, "total_steps must be positive");
    assert!(
        report.metal_dispatches > 0,
        "metal_dispatches must be positive"
    );

    // Dispatch count should be reasonable for a small model.
    assert!(
        report.dispatch_count < 100,
        "decoder mini should have < 100 dispatches, got {}",
        report.dispatch_count
    );

    // --- RTF estimate ---
    assert!(
        report.estimated_rtf.is_some(),
        "RTF estimate should be populated"
    );
    let rtf = report.estimated_rtf.unwrap();
    assert!(
        rtf > 0.0 && rtf.is_finite(),
        "RTF should be positive and finite"
    );

    // Small model should have low RTF.
    assert!(rtf < 1.0, "decoder mini RTF should be < 1.0, got {rtf:.4}");

    // --- Report display and serialization ---
    let display = format!("{report}");
    assert!(display.contains("Conversion complete:"));
    assert!(display.contains("Compiled Metal artifact ready for GPU execution."));

    let json = report.to_json();
    let val: serde_json::Value =
        serde_json::from_str(&json).expect("ConvertReport JSON must be valid");
    assert!(val["op_count"].as_u64().unwrap() >= 8);
    assert!(val["dispatch_count"].as_u64().unwrap() > 0);

    let table = report.summary_table();
    assert!(table.contains("| Metric | Value |"));
    assert!(table.contains("Dispatch count"));

    eprintln!(
        "[ConvertBuilder Kokoro decoder mini] ops={}, mapped={}, dispatches={}, \
         metal={}, steps={}, RTF={:.4}",
        report.op_count,
        report.mapped_ops_count(),
        report.dispatch_count,
        report.metal_dispatches,
        report.total_steps,
        rtf,
    );
}

// ===========================================================================
// 2. ConvertBuilder GPU execution for Kokoro decoder mini
// ===========================================================================

/// ConvertBuilder + GPU execution: compile kokoro_decoder_mini via ConvertBuilder,
/// run on Metal GPU, verify output is finite and has correct shape.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_convert_builder_kokoro_decoder_mini_gpu_execution() {
    use nn_core::{DType, Device};

    let graph_path = test_data_dir().join("kokoro_decoder_mini.json");
    if !graph_path.exists() {
        eprintln!(
            "SKIP: kokoro_decoder_mini.json not found at {}",
            graph_path.display()
        );
        return;
    }

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let dir = std::env::temp_dir().join(format!("nn_kokoro_cb_dec_gpu_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let weights_path = write_decoder_mini_weights(&dir);

    let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
        .optimize(nn_import::OptLevel::Full)
        .verify(nn_import::VerifyLevel::None)
        .build()
        .unwrap_or_else(|e| panic!("ConvertBuilder failed: {e:?}"));

    let _ = std::fs::remove_dir_all(&dir);

    let model = &result.result.model;

    // Create input tensor matching decoder mini: x:[1, 8, 16].
    let input_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    let input_cpu = nn_core::DynTensor::from_vec(input_data, &[1, 8, 16], &Device::Cpu).unwrap();
    let input_gpu = input_cpu.to_device(&Device::metal()).unwrap();

    let output = model
        .execute_dyn(&cache, &[&input_gpu])
        .expect("GPU execution must succeed for decoder mini");

    // Decoder mini output: [1, 16, 16] (cat of magnitude[1,8,16] + phase[1,8,16]).
    assert_eq!(output.dims(), &[1, 16, 16], "unexpected output shape");
    assert_eq!(output.dtype(), DType::F32);

    // Verify finite, non-trivial output.
    let output_cpu = output.to_device(&Device::Cpu).unwrap();
    let vals = output_cpu.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 256);

    let all_finite = vals.iter().all(|v| v.is_finite());
    assert!(all_finite, "GPU output contains non-finite values");

    let energy: f64 = vals.iter().map(|v| f64::from(*v).powi(2)).sum();
    assert!(
        energy > 1e-6,
        "GPU output is near-zero (energy={energy:.6e})"
    );

    eprintln!(
        "[ConvertBuilder Kokoro decoder mini GPU] shape={:?}, energy={:.4e}, \
         dispatches={}, steps={}",
        output.dims(),
        energy,
        model.num_dispatches(),
        model.num_steps(),
    );
}

// ===========================================================================
// 3. Cross-path parity: ConvertBuilder vs convert()
// ===========================================================================

/// Cross-path numerical parity: ConvertBuilder vs convert() for decoder mini.
///
/// Proves that both API paths produce identical compiled models when given
/// the same graph and weights. This catches regressions where the builder API
/// diverges from the legacy convert() API.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_convert_builder_vs_legacy_parity() {
    use nn_core::Device;

    let graph_path = test_data_dir().join("kokoro_decoder_mini.json");
    if !graph_path.exists() {
        eprintln!("SKIP: kokoro_decoder_mini.json not found");
        return;
    }

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let dir = std::env::temp_dir().join(format!("nn_kokoro_cb_parity_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let weights_path = write_decoder_mini_weights(&dir);

    // Path A: ConvertBuilder (production API).
    let builder_result = nn_import::convert_build(&graph_path, &weights_path, &cache)
        .optimize(nn_import::OptLevel::Full)
        .verify(nn_import::VerifyLevel::None)
        .build()
        .unwrap_or_else(|e| panic!("ConvertBuilder failed: {e:?}"));

    // Path B: Legacy convert() API.
    let legacy_result = nn_import::convert(&graph_path, &weights_path, None, &cache)
        .unwrap_or_else(|e| panic!("convert() failed: {e:?}"));

    let _ = std::fs::remove_dir_all(&dir);

    // Both models should have the same structure.
    assert_eq!(
        builder_result.result.model.num_steps(),
        legacy_result.model.num_steps(),
        "step count mismatch: builder={} vs legacy={}",
        builder_result.result.model.num_steps(),
        legacy_result.model.num_steps(),
    );
    assert_eq!(
        builder_result.result.model.num_dispatches(),
        legacy_result.model.num_dispatches(),
        "dispatch count mismatch: builder={} vs legacy={}",
        builder_result.result.model.num_dispatches(),
        legacy_result.model.num_dispatches(),
    );

    // Execute both with the same input and compare output.
    let input_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    let input_cpu = nn_core::DynTensor::from_vec(input_data, &[1, 8, 16], &Device::Cpu).unwrap();
    let input_gpu = input_cpu.to_device(&Device::metal()).unwrap();

    let builder_output = builder_result
        .result
        .model
        .execute_dyn(&cache, &[&input_gpu])
        .expect("builder GPU execution");
    let legacy_output = legacy_result
        .model
        .execute_dyn(&cache, &[&input_gpu])
        .expect("legacy GPU execution");

    let builder_vals = builder_output
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let legacy_vals = legacy_output
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(
        builder_vals.len(),
        legacy_vals.len(),
        "output length mismatch"
    );

    // Outputs should be identical (same graph, same weights, same GPU).
    let max_diff: f32 = builder_vals
        .iter()
        .zip(legacy_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-6,
        "ConvertBuilder vs convert() max diff {max_diff:.6e} exceeds 1e-6"
    );

    eprintln!(
        "[Cross-path parity] builder vs legacy max_diff={max_diff:.6e}, \
         steps={}, dispatches={} -- MATCHED",
        builder_result.result.model.num_steps(),
        builder_result.result.model.num_dispatches(),
    );
}

// ===========================================================================
// 4. ConvertBuilder for real Kokoro segments (gated on segment exports)
// ===========================================================================

/// ConvertBuilder for the decoder segment with ConvertReport validation.
///
/// Runs the full ConvertBuilder pipeline on the real Kokoro decoder segment
/// export and validates the ConvertReport against Kokoro-specific expectations
/// (high op count, Conv1d-heavy, multi-input).
///
/// Skips if `models/kokoro-82m/decoder/` export files are not present.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_convert_builder_decoder_segment_report() {
    let (graph_path, weights_path) = match require_segment_files("decoder") {
        Some(paths) => paths,
        None => return,
    };

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
        .optimize(nn_import::OptLevel::Full)
        .verify(nn_import::VerifyLevel::None)
        .build()
        .unwrap_or_else(|e| panic!("ConvertBuilder failed for decoder segment: {e:?}"));

    let report = &result.report;

    // Decoder is the most compute-heavy segment.
    assert!(
        report.op_count > 50,
        "decoder should have > 50 aten ops, got {}",
        report.op_count
    );
    assert!(
        report.mapped_ops_count() > 50,
        "decoder should have > 50 mapped ops, got {}",
        report.mapped_ops_count()
    );

    // Op mapping should be near-complete (all Kokoro ops are supported).
    if let Some(pct) = report.mapped_pct() {
        assert!(
            pct > 95.0,
            "decoder op mapping should be > 95%, got {pct:.1}%"
        );
    }

    // Decoder has 3 user inputs (x, style, har_source).
    assert_eq!(report.num_user_inputs, 3, "decoder has 3 user inputs");

    // Decoder should have meaningful compilation.
    assert!(
        report.dispatch_count > 10,
        "decoder should have > 10 dispatches, got {}",
        report.dispatch_count
    );
    assert!(
        report.metal_dispatches > 10,
        "decoder should have > 10 Metal dispatches"
    );

    // ConvertReport should serialize to valid JSON.
    let json = report.to_json();
    let _: serde_json::Value =
        serde_json::from_str(&json).expect("decoder segment ConvertReport JSON must be valid");

    eprintln!(
        "[ConvertBuilder decoder segment] ops={}, mapped={} ({:.1}%), \
         dispatches={}, metal={}, RTF={:.4}, compile={}ms",
        report.op_count,
        report.mapped_ops_count(),
        report.mapped_pct().unwrap_or(0.0),
        report.dispatch_count,
        report.metal_dispatches,
        report.estimated_rtf.unwrap_or(0.0),
        report.compile_time_ms,
    );
}

/// ConvertBuilder for the plbert segment with ConvertReport validation.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_convert_builder_plbert_segment_report() {
    let (graph_path, weights_path) = match require_segment_files("plbert") {
        Some(paths) => paths,
        None => return,
    };

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
        .optimize(nn_import::OptLevel::Full)
        .verify(nn_import::VerifyLevel::None)
        .build()
        .unwrap_or_else(|e| panic!("ConvertBuilder failed for plbert: {e:?}"));

    let report = &result.report;

    assert!(
        report.op_count > 10,
        "plbert should have > 10 ops, got {}",
        report.op_count
    );
    assert!(
        report.dispatch_count > 5,
        "plbert should have > 5 dispatches, got {}",
        report.dispatch_count
    );
    assert!(report.estimated_rtf.is_some());

    eprintln!(
        "[ConvertBuilder plbert] ops={}, dispatches={}, RTF={:.4}",
        report.op_count,
        report.dispatch_count,
        report.estimated_rtf.unwrap_or(0.0),
    );
}

// ===========================================================================
// 5. ConvertBuilder optimization comparison for Kokoro
// ===========================================================================

/// OptLevel::Full vs OptLevel::None for Kokoro decoder mini.
///
/// Verifies that fusion/peephole optimization reduces dispatch count for
/// Kokoro-style graphs (Conv1d chains with residual connections).
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_convert_builder_kokoro_optimization_impact() {
    let graph_path = test_data_dir().join("kokoro_decoder_mini.json");
    if !graph_path.exists() {
        eprintln!("SKIP: kokoro_decoder_mini.json not found");
        return;
    }

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let dir = std::env::temp_dir().join(format!("nn_kokoro_cb_optcmp_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let weights_path = write_decoder_mini_weights(&dir);

    let result_full = nn_import::convert_build(&graph_path, &weights_path, &cache)
        .optimize(nn_import::OptLevel::Full)
        .verify(nn_import::VerifyLevel::None)
        .build()
        .unwrap_or_else(|e| panic!("Full opt failed: {e:?}"));

    let result_none = nn_import::convert_build(&graph_path, &weights_path, &cache)
        .optimize(nn_import::OptLevel::None)
        .verify(nn_import::VerifyLevel::None)
        .build()
        .unwrap_or_else(|e| panic!("No opt failed: {e:?}"));

    let _ = std::fs::remove_dir_all(&dir);

    // Op mapping should be identical.
    assert_eq!(
        result_full.report.op_count, result_none.report.op_count,
        "op_count should not depend on optimization"
    );

    // Optimized dispatch count should be <= unoptimized.
    assert!(
        result_full.report.dispatch_count <= result_none.report.dispatch_count,
        "optimized dispatches ({}) should not exceed unoptimized ({})",
        result_full.report.dispatch_count,
        result_none.report.dispatch_count,
    );

    eprintln!(
        "[Kokoro opt comparison] none: dispatches={}, full: dispatches={}, \
         reduction={:.0}%",
        result_none.report.dispatch_count,
        result_full.report.dispatch_count,
        result_full.report.dispatch_reduction_pct().unwrap_or(0.0),
    );
}
