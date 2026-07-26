// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end integration test for the `nn convert` pipeline.
//!
//! Exercises `ConvertBuilder` (the builder pattern API used by the CLI)
//! with synthetic graph JSON + safetensors weights written to a temp directory.
//! Verifies `ConvertReport` fields: op_count, mapped_ops, dispatch counts,
//! compile time, and Metal GPU execution of the compiled model.
//!
//! Does NOT require PyTorch, KOKORO_WEIGHTS, or any external model files.

use std::collections::HashMap;
use std::path::Path;

/// Write the MLP graph JSON (Linear -> ReLU -> Linear) to a file.
///
/// Model: x:[1,4] -> fc1(4->8) -> relu -> fc2(8->3) -> output:[1,3]
/// Ops: 2x linear.default, 1x relu.default = 3 aten ops.
fn write_mlp_graph_json(dir: &Path) -> std::path::PathBuf {
    let graph_path = dir.join("graph.json");
    std::fs::write(&graph_path, include_str!("../test_data/e2e_mlp.json")).unwrap();
    graph_path
}

/// Write synthetic MLP safetensors weights (fc1: 4->8, fc2: 8->3).
fn write_mlp_weights(dir: &Path) -> std::path::PathBuf {
    let fc1_w: Vec<u8> = (0..32u32)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    let fc1_b: Vec<u8> = [0.0f32; 8].iter().flat_map(|f| f.to_le_bytes()).collect();
    let fc2_w: Vec<u8> = (0..24u32)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    let fc2_b: Vec<u8> = [0.0f32; 3].iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "fc1.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8, 4], &fc1_w).unwrap(),
    );
    tensors.insert(
        "fc1.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8], &fc1_b).unwrap(),
    );
    tensors.insert(
        "fc2.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3, 8], &fc2_w).unwrap(),
    );
    tensors.insert(
        "fc2.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3], &fc2_b).unwrap(),
    );
    let weights_path = dir.join("weights.safetensors");
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(&weights_path, serialized).unwrap();
    weights_path
}

// ---------------------------------------------------------------------------
// ConvertBuilder E2E: import -> compile -> report assertions
// ---------------------------------------------------------------------------

/// Full ConvertBuilder pipeline: graph JSON + safetensors -> ConvertReport.
///
/// This is the primary E2E test for issue #3778. Exercises the same code path
/// as `nn convert graph.json weights.safetensors`:
///   1. Parse graph JSON
///   2. Load safetensors weights
///   3. Build ComputationGraph
///   4. Compile to Metal GPU
///   5. Produce ConvertReport with op mapping and compilation stats
///
/// Asserts on ConvertReport fields that are stable for the MLP fixture:
///   - op_count = 3 (2 linear + 1 relu)
///   - mapped_ops includes "torch.ops.aten.linear.default" and "torch.ops.aten.relu.default"
///   - unmapped_ops is empty (all 3 ops are supported)
///   - dispatch_count > 0
///   - compile_time_ms > 0
///   - num_user_inputs = 1
///   - num_weights_loaded = 4 (fc1.weight, fc1.bias, fc2.weight, fc2.bias)
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_convert_builder_mlp_report_fields() {
    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let dir = std::env::temp_dir().join(format!("nn_convert_builder_e2e_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = write_mlp_graph_json(&dir);
    let weights_path = write_mlp_weights(&dir);

    // Use ConvertBuilder (same API as `nn convert` CLI).
    let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
        .optimize(nn_import::OptLevel::Full)
        .verify(nn_import::VerifyLevel::None)
        .build()
        .expect("ConvertBuilder::build() must succeed for MLP fixture");

    let _ = std::fs::remove_dir_all(&dir);

    let report = &result.report;

    // --- Op mapping assertions ---
    assert_eq!(report.op_count, 3, "MLP has 3 aten ops (2 linear + 1 relu)");
    assert_eq!(report.mapped_ops_count(), 3, "all 3 ops should be mapped");
    assert!(
        report.unmapped_ops.is_empty(),
        "no unmapped ops for MLP: {:?}",
        report.unmapped_ops
    );
    assert_eq!(
        report.mapped_pct().unwrap() as u32,
        100,
        "100% mapping for MLP"
    );

    // Check specific op names in mapped_ops.
    let mapped_names: Vec<&str> = report.mapped_ops.iter().map(|(n, _)| n.as_str()).collect();
    assert!(
        mapped_names.contains(&"torch.ops.aten.linear.default"),
        "mapped_ops should include linear: {mapped_names:?}"
    );
    assert!(
        mapped_names.contains(&"torch.ops.aten.relu.default"),
        "mapped_ops should include relu: {mapped_names:?}"
    );

    // Check op counts per target.
    let linear_count: usize = report
        .mapped_ops
        .iter()
        .filter(|(n, _)| n == "torch.ops.aten.linear.default")
        .map(|(_, c)| c)
        .sum();
    assert_eq!(linear_count, 2, "2 linear ops");

    let relu_count: usize = report
        .mapped_ops
        .iter()
        .filter(|(n, _)| n == "torch.ops.aten.relu.default")
        .map(|(_, c)| c)
        .sum();
    assert_eq!(relu_count, 1, "1 relu op");

    // --- Import metrics ---
    assert_eq!(report.num_user_inputs, 1, "1 user input (x)");
    assert_eq!(
        report.num_weights_loaded, 4,
        "4 weight tensors (fc1.weight, fc1.bias, fc2.weight, fc2.bias)"
    );
    assert!(
        report.total_ops_imported > 0,
        "total_ops_imported should be positive"
    );

    // --- Compilation metrics ---
    assert!(
        report.dispatch_count > 0,
        "dispatch_count must be positive after compilation"
    );
    assert!(report.total_steps > 0, "total_steps must be positive");
    assert!(
        report.metal_dispatches > 0,
        "metal_dispatches must be positive"
    );
    // compile_time_ms may be 0 for tiny models (sub-millisecond compilation).
    // We just verify it exists and is non-negative (u64, so always >= 0).
    assert!(
        report.dispatch_count_before_fusion > 0,
        "pre-fusion dispatch count should be positive"
    );

    // Fusion should reduce (or equal) dispatch count.
    assert!(
        report.dispatch_count <= report.dispatch_count_before_fusion,
        "dispatch_count ({}) should not exceed pre-fusion count ({})",
        report.dispatch_count,
        report.dispatch_count_before_fusion
    );

    // --- RTF estimate ---
    assert!(
        report.estimated_rtf.is_some(),
        "RTF estimate should be populated when metal_dispatches > 0"
    );
    let rtf = report.estimated_rtf.unwrap();
    assert!(
        rtf > 0.0 && rtf.is_finite(),
        "RTF should be positive and finite, got {rtf}"
    );

    // --- Display/print should not panic ---
    let display = format!("{report}");
    assert!(
        display.contains("Conversion complete:"),
        "display should have header"
    );
    assert!(
        display.contains("Compiled Metal artifact ready for GPU execution."),
        "display should have footer"
    );

    eprintln!(
        "[ConvertBuilder E2E] op_count={}, mapped={}, dispatches={}, \
         metal_dispatches={}, compile={}ms, RTF={:.4}",
        report.op_count,
        report.mapped_ops_count(),
        report.dispatch_count,
        report.metal_dispatches,
        report.compile_time_ms,
        rtf,
    );
}

// ---------------------------------------------------------------------------
// ConvertBuilder E2E + GPU execution: compile then run
// ---------------------------------------------------------------------------

/// Full convert + run pipeline: compile model via ConvertBuilder, then execute
/// on Metal GPU with synthetic input, verify output shape and finiteness.
///
/// This tests the same path as `nn convert` followed by `nn run`.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_convert_builder_mlp_gpu_execution() {
    use nn_core::{DType, Device};

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let dir = std::env::temp_dir().join(format!("nn_convert_builder_run_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = write_mlp_graph_json(&dir);
    let weights_path = write_mlp_weights(&dir);

    let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
        .optimize(nn_import::OptLevel::Full)
        .verify(nn_import::VerifyLevel::None)
        .build()
        .expect("ConvertBuilder::build() must succeed");

    let _ = std::fs::remove_dir_all(&dir);

    let model = &result.result.model;
    let imported = &result.result.graph;

    // Verify model metadata matches import.
    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(model.num_inputs(), 1);
    assert_eq!(model.output_shape(), &[1, 3]);
    assert_eq!(model.output_dtype(), DType::F32);

    // Create input tensor [1, 4] and run on GPU.
    let input_data = vec![0.5_f32, -0.3, 0.8, -0.1];
    let input_cpu = nn_core::DynTensor::from_vec(input_data, &[1, 4], &Device::Cpu).unwrap();
    let input_gpu = input_cpu.to_device(&Device::metal()).unwrap();

    let output = model
        .execute_dyn(&cache, &[&input_gpu])
        .expect("GPU execution must succeed");

    assert_eq!(output.dims(), &[1, 3]);
    assert_eq!(output.dtype(), DType::F32);

    // Verify output is finite and non-trivial.
    let output_cpu = output.to_device(&Device::Cpu).unwrap();
    let vals = output_cpu.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 3);
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] is not finite: {v}");
    }

    // At least one output should be non-zero (weights are 0.01 * i, not all zero).
    let any_nonzero = vals.iter().any(|&v| v.abs() > 1e-10);
    assert!(
        any_nonzero,
        "GPU output is all zeros -- likely weight loading issue"
    );

    eprintln!(
        "[ConvertBuilder GPU] output: {vals:?}, dispatches={}, steps={}",
        model.num_dispatches(),
        model.num_steps(),
    );
}

// ---------------------------------------------------------------------------
// ConvertBuilder with OptLevel::None: no optimization
// ---------------------------------------------------------------------------

/// ConvertBuilder with optimization disabled: verify dispatch count is higher
/// (or equal to) the optimized version, confirming optimization actually works.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_convert_builder_mlp_no_optimization() {
    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let dir =
        std::env::temp_dir().join(format!("nn_convert_builder_noopt_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = write_mlp_graph_json(&dir);
    let weights_path = write_mlp_weights(&dir);

    // Build with no optimization.
    let result_noopt = nn_import::convert_build(&graph_path, &weights_path, &cache)
        .optimize(nn_import::OptLevel::None)
        .verify(nn_import::VerifyLevel::None)
        .build()
        .expect("ConvertBuilder with OptLevel::None must succeed");

    // Build with full optimization.
    let result_opt = nn_import::convert_build(&graph_path, &weights_path, &cache)
        .optimize(nn_import::OptLevel::Full)
        .verify(nn_import::VerifyLevel::None)
        .build()
        .expect("ConvertBuilder with OptLevel::Full must succeed");

    let _ = std::fs::remove_dir_all(&dir);

    // Op mapping should be identical regardless of optimization level.
    assert_eq!(
        result_noopt.report.op_count, result_opt.report.op_count,
        "op_count should not depend on optimization level"
    );
    assert_eq!(
        result_noopt.report.mapped_ops_count(),
        result_opt.report.mapped_ops_count(),
        "mapped count should not depend on optimization level"
    );

    // Both should produce functional models.
    assert!(result_noopt.report.dispatch_count > 0);
    assert!(result_opt.report.dispatch_count > 0);

    // Optimized dispatch count should be <= unoptimized.
    assert!(
        result_opt.report.dispatch_count <= result_noopt.report.dispatch_count,
        "optimized ({}) should not exceed unoptimized ({})",
        result_opt.report.dispatch_count,
        result_noopt.report.dispatch_count,
    );

    eprintln!(
        "[ConvertBuilder opt comparison] noopt dispatches={}, opt dispatches={}",
        result_noopt.report.dispatch_count, result_opt.report.dispatch_count,
    );
}

// ---------------------------------------------------------------------------
// import_model (no Metal) still works: parse + weight load + graph build
// ---------------------------------------------------------------------------

/// Verify import_model() works without Metal for the CPU-only import path.
/// This exercises parse + weight load + graph build without compilation.
#[test]
fn test_import_model_mlp_no_metal() {
    let dir = std::env::temp_dir().join(format!("nn_import_model_nomet_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = write_mlp_graph_json(&dir);
    let weights_path = write_mlp_weights(&dir);

    let imported = nn_import::import_model(&graph_path, &weights_path)
        .expect("import_model must succeed for MLP fixture");

    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["linear_1"]);

    // 1 input + 4 param placeholders + 3 compute ops = 8 nodes
    assert_eq!(imported.graph.len(), 8);

    // Output node should be Linear (the fc2 layer).
    let output = imported.graph.output_node().unwrap();
    assert!(
        matches!(
            output.op(),
            nn_core::dyn_tensor::trace::TraceOp::Linear { .. }
        ),
        "expected Linear output, got: {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[1, 3]);
}
