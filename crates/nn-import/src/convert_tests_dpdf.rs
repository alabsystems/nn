// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! dpdf backbone convert tests: Conv2d-BN-SiLU backbone + View + Linear.
//!
//! Models the first two stages of a DocLayout-YOLO backbone, exercising the
//! vision-model ops that appear in dpdf document processing models:
//! Conv2d (stride 2) -> BatchNorm -> SiLU -> Conv2d -> BatchNorm -> SiLU ->
//! View (flatten) -> Linear.
//!
//! Part of #2293 (nn::convert() automatic PyTorch model porting).

use std::collections::HashMap;
use std::path::Path;

use nn_core::dyn_tensor::trace::TraceOp;

use crate::graph_build::ImportedGraph;
use crate::import_model;

#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::convert;

/// Write synthetic dpdf backbone weights to a safetensors file.
///
/// Stage 0: Conv2d [16, 3, 3, 3] = 432 elements, bias [16]
///          BN weight [16], bias [16], running_mean [16], running_var [16]
/// Stage 1: Conv2d [32, 16, 3, 3] = 4608 elements, bias [32]
///          BN weight [32], bias [32], running_mean [32], running_var [32]
/// Head:    Linear weight [10, 2048] = 20480, bias [10]
fn write_dpdf_backbone_weights(dir: &Path) -> std::path::PathBuf {
    let mut tensors = HashMap::new();

    // Stage 0 conv: [16, 3, 3, 3] = 432 elements
    let conv1_w: Vec<u8> = (0..432)
        .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
        .collect();
    let conv1_b: Vec<u8> = [0.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn1_w: Vec<u8> = [1.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn1_b: Vec<u8> = [0.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn1_mean: Vec<u8> = [0.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn1_var: Vec<u8> = [1.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();

    // Stage 1 conv: [32, 16, 3, 3] = 4608 elements
    let conv2_w: Vec<u8> = (0..4608)
        .flat_map(|i| ((i as f32) * 0.0001).to_le_bytes())
        .collect();
    let conv2_b: Vec<u8> = [0.0f32; 32].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn2_w: Vec<u8> = [1.0f32; 32].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn2_b: Vec<u8> = [0.0f32; 32].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn2_mean: Vec<u8> = [0.0f32; 32].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn2_var: Vec<u8> = [1.0f32; 32].iter().flat_map(|f| f.to_le_bytes()).collect();

    // Head linear: [10, 2048] = 20480 elements
    let fc_w: Vec<u8> = (0..20480)
        .flat_map(|i| ((i as f32) * 0.0001).to_le_bytes())
        .collect();
    let fc_b: Vec<u8> = [0.0f32; 10].iter().flat_map(|f| f.to_le_bytes()).collect();

    // Register all tensors with FQN keys matching the signature.
    for (name, shape, data) in [
        (
            "backbone.stage0.conv.weight",
            vec![16, 3, 3, 3],
            conv1_w.as_slice(),
        ),
        ("backbone.stage0.conv.bias", vec![16], conv1_b.as_slice()),
        ("backbone.stage0.bn.weight", vec![16], bn1_w.as_slice()),
        ("backbone.stage0.bn.bias", vec![16], bn1_b.as_slice()),
        (
            "backbone.stage0.bn.running_mean",
            vec![16],
            bn1_mean.as_slice(),
        ),
        (
            "backbone.stage0.bn.running_var",
            vec![16],
            bn1_var.as_slice(),
        ),
        (
            "backbone.stage1.conv.weight",
            vec![32, 16, 3, 3],
            conv2_w.as_slice(),
        ),
        ("backbone.stage1.conv.bias", vec![32], conv2_b.as_slice()),
        ("backbone.stage1.bn.weight", vec![32], bn2_w.as_slice()),
        ("backbone.stage1.bn.bias", vec![32], bn2_b.as_slice()),
        (
            "backbone.stage1.bn.running_mean",
            vec![32],
            bn2_mean.as_slice(),
        ),
        (
            "backbone.stage1.bn.running_var",
            vec![32],
            bn2_var.as_slice(),
        ),
        ("head.fc.weight", vec![10, 2048], fc_w.as_slice()),
        ("head.fc.bias", vec![10], fc_b.as_slice()),
    ] {
        tensors.insert(
            name.to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, data).unwrap(),
        );
    }

    let weights_path = dir.join("weights.safetensors");
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(&weights_path, serialized).unwrap();
    weights_path
}

/// Import the dpdf backbone mini fixture from disk.
fn import_dpdf_backbone_fixture() -> ImportedGraph {
    // Several tests call this helper and run in parallel within one process.
    // The temp dir must be unique per call, otherwise one test's cleanup
    // (remove_dir_all below) races with another test's read of the same path.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "nn_import_dpdf_{}_{}",
        std::process::id(),
        unique
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../test_data/dpdf_backbone_mini.json"),
    )
    .unwrap();
    let weights_path = write_dpdf_backbone_weights(&dir);
    let imported = import_model(&graph_path, &weights_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    imported
}

// ---------------------------------------------------------------------------
// Graph structure tests (no Metal required)
// ---------------------------------------------------------------------------

/// E2E: dpdf backbone imports with correct structure.
///
/// Exercises the full import pipeline: parse JSON -> weight load -> FQN mapping ->
/// op mapping -> graph build -> topology validation.
///
/// This models the first two stages of DocLayout-YOLO backbone:
/// Conv2d(3->16, stride=2) -> BN -> SiLU -> Conv2d(16->32, stride=2) -> BN ->
/// SiLU -> View (flatten) -> Linear(2048->10)
#[test]
fn test_import_dpdf_backbone_structure() {
    let imported = import_dpdf_backbone_fixture();

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["linear"]);

    // 1 Input + 14 params/buffers + 8 compute ops = 23 total nodes.
    assert_eq!(imported.graph.len(), 23);

    let output = imported.graph.output_node().unwrap();
    assert!(
        matches!(output.op(), TraceOp::Linear { .. }),
        "expected Linear as output, got: {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[1, 10]);
}

/// E2E: all dpdf-specific aten ops map to correct TraceOp variants.
#[test]
fn test_import_dpdf_backbone_op_counts() {
    let imported = import_dpdf_backbone_fixture();
    let nodes = imported.graph.nodes();
    let count = |pred: fn(&TraceOp) -> bool| nodes.iter().filter(|n| pred(n.op())).count();

    assert_eq!(count(|op| matches!(op, TraceOp::Conv2d { .. })), 2);
    assert_eq!(count(|op| matches!(op, TraceOp::BatchNorm { .. })), 2);
    assert_eq!(count(|op| matches!(op, TraceOp::Silu)), 2);
    assert_eq!(count(|op| matches!(op, TraceOp::Reshape { .. })), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::Linear { .. })), 1);
}

/// E2E: intermediate shapes propagate correctly through stride-2 convolutions.
#[test]
fn test_import_dpdf_backbone_shapes() {
    let imported = import_dpdf_backbone_fixture();
    let nodes = imported.graph.nodes();

    // Input: [1, 3, 32, 32]
    let input = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Input))
        .unwrap();
    assert_eq!(input.output_shape(), &[1, 3, 32, 32]);

    // Conv1 output: [1, 16, 16, 16] (stride=2, padding=1)
    let conv1 = nodes.iter().find(|n| n.name() == "conv1").unwrap();
    assert_eq!(conv1.output_shape(), &[1, 16, 16, 16]);

    // Conv2 output: [1, 32, 8, 8] (stride=2, padding=1)
    let conv2 = nodes.iter().find(|n| n.name() == "conv2").unwrap();
    assert_eq!(conv2.output_shape(), &[1, 32, 8, 8]);

    // Flatten output: [1, 2048] (32 * 8 * 8 = 2048)
    let flat = nodes.iter().find(|n| n.name() == "flat").unwrap();
    assert_eq!(flat.output_shape(), &[1, 2048]);

    // Linear output: [1, 10]
    let linear = nodes.iter().find(|n| n.name() == "linear").unwrap();
    assert_eq!(linear.output_shape(), &[1, 10]);
}

/// E2E: BatchNorm parameters (eps) survive import.
#[test]
fn test_import_dpdf_backbone_bn_params() {
    let imported = import_dpdf_backbone_fixture();
    let nodes = imported.graph.nodes();

    let bn_nodes: Vec<_> = nodes
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::BatchNorm { .. }))
        .collect();
    assert_eq!(bn_nodes.len(), 2);

    for bn in &bn_nodes {
        if let TraceOp::BatchNorm { eps, .. } = bn.op() {
            assert!((*eps - 1e-5).abs() < 1e-8, "expected eps=1e-5, got {eps}");
        }
    }
}

// ---------------------------------------------------------------------------
// Verification bridge tests (require `verify` feature)
// ---------------------------------------------------------------------------

/// E2E: imported dpdf backbone -> NY IBP via shaped bounds.
///
/// Exercises: parse JSON -> build ComputationGraph -> translate to NY
/// GraphNetwork -> IBP propagation -> finite output bounds.
///
/// This is the L2 (composition bounds) proof layer of nn::convert() for
/// vision-model (Conv2d + BN + SiLU) graphs.
#[test]
#[cfg(feature = "verify")]
fn test_import_dpdf_backbone_ibp_bounds() {
    use ndarray::{ArrayD, IxDyn};

    let imported = import_dpdf_backbone_fixture();

    let gn = nn_verify::trace_to_graph_model(&imported.graph)
        .expect("trace_to_graph_model must succeed for dpdf backbone")
        .graph;

    let input_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Input))
        .expect("must have Input node");
    let shape = input_node.output_shape();
    assert_eq!(shape, &[1, 3, 32, 32], "dpdf backbone input shape");

    let lower = ArrayD::from_elem(IxDyn(shape), -1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(shape), 1.0_f32);
    let input_bounds = nn_verify::BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = gn
        .propagate_ibp(&input_bounds)
        .expect("IBP propagation must succeed for dpdf backbone");

    let (out_lo, out_hi) = output.lower_upper();

    for (idx, (&lo, &hi)) in out_lo.iter().zip(out_hi.iter()).enumerate() {
        assert!(lo.is_finite(), "lower bound at idx {idx} not finite: {lo}");
        assert!(hi.is_finite(), "upper bound at idx {idx} not finite: {hi}");
        assert!(lo <= hi, "bounds inverted at idx {idx}: lo={lo} > hi={hi}");
    }

    let max_width = out_hi
        .iter()
        .zip(out_lo.iter())
        .map(|(hi, lo)| hi - lo)
        .fold(0.0_f32, f32::max);
    assert!(
        max_width > 0.0,
        "IBP bounds should be non-trivial, max_width={max_width}"
    );

    eprintln!(
        "[dpdf backbone IBP] {} output elements, max_width={max_width:.4}",
        out_lo.len()
    );
}

// ---------------------------------------------------------------------------
// Metal GPU execution tests (require `metal` feature + macOS)
// ---------------------------------------------------------------------------

/// Import -> CompiledModel -> execute_dyn on Metal GPU for the dpdf backbone.
///
/// Exercises the full convert pipeline: parse JSON -> build ComputationGraph ->
/// trace compile -> Metal pipeline creation -> GPU forward -> finite output.
///
/// This validates Conv2d (stride=2) + BatchNorm + SiLU + View (flatten) +
/// Linear all compile and execute correctly on Metal GPU.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_import_compile_execute_dpdf_backbone_gpu() {
    use nn_core::{DType, Device};
    use nn_metal::compiled_model::CompiledModel;

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let imported = import_dpdf_backbone_fixture();

    let compiled = CompiledModel::builder(&imported.graph, &cache)
        .build()
        .expect("compile dpdf backbone to Metal");

    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.output_shape(), &[1, 10]);
    assert_eq!(compiled.output_dtype(), DType::F32);

    let nd = compiled.num_dispatches();
    assert!(nd >= 1, "expected at least 1 dispatch, got {nd}");
    eprintln!(
        "[dpdf backbone GPU] steps={}, dispatches={nd}",
        compiled.num_steps()
    );

    // Create GPU input: [1, 3, 32, 32] with deterministic values (image-like).
    let input_size = 3 * 32 * 32;
    let input_data: Vec<f32> = (0..input_size)
        .map(|i| (i as f32) / input_size as f32)
        .collect();
    let input_cpu =
        nn_core::DynTensor::from_vec(input_data, &[1, 3, 32, 32], &Device::Cpu).unwrap();
    let input_gpu = input_cpu.to_device(&Device::metal()).unwrap();

    let output = compiled
        .execute_dyn(&cache, &[&input_gpu])
        .expect("GPU execution must succeed");

    assert_eq!(output.dims(), &[1, 10]);
    assert_eq!(output.dtype(), DType::F32);

    let output_cpu = output.to_device(&Device::Cpu).unwrap();
    let vals = output_cpu.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 10);
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.is_finite(),
            "dpdf backbone GPU output[{i}] is not finite: {v}"
        );
    }

    let any_nonzero = vals.iter().any(|&v| v.abs() > 1e-10);
    assert!(
        any_nonzero,
        "GPU output is all zeros -- likely weight loading issue"
    );

    eprintln!(
        "[dpdf backbone GPU] output: {} elements, range=[{:.4}, {:.4}]",
        vals.len(),
        vals.iter().copied().fold(f32::INFINITY, f32::min),
        vals.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    );
}

/// Full convert() pipeline: dpdf backbone import + compile + proof chain.
///
/// Exercises the top-level `convert()` API with a dpdf-style vision model.
/// Validates all three proof layers:
///   L0: GPU execution produces finite output
///   L1: Kani (None -- populated by Prover separately)
///   L2: IBP composition bounds (if `verify` feature is on)
///   L3: Reference parity (None -- no reference trace provided)
///
/// Part of #2293 (nn::convert() automatic model porting).
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_convert_dpdf_backbone_full_pipeline() {
    use nn_core::{DType, Device};

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let dir = std::env::temp_dir().join(format!("nn_convert_dpdf_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../test_data/dpdf_backbone_mini.json"),
    )
    .unwrap();
    let weights_path = write_dpdf_backbone_weights(&dir);

    let result = convert(&graph_path, &weights_path, None, &cache).expect("convert() must succeed");
    let _ = std::fs::remove_dir_all(&dir);

    // L1/L3: absent (no Kani, no reference trace).
    assert!(result.proof.kernel_safety.is_none());
    assert!(result.proof.reference_parity.is_none());

    // Graph metadata.
    assert_eq!(result.graph.num_user_inputs, 1);
    assert_eq!(result.graph.user_input_names, vec!["x"]);
    assert_eq!(result.graph.output_names, vec!["linear"]);

    // L0: GPU execution produces 10-class logits.
    let input_size = 3 * 32 * 32;
    let input_data: Vec<f32> = (0..input_size)
        .map(|i| (i as f32) / input_size as f32)
        .collect();
    let input_cpu =
        nn_core::DynTensor::from_vec(input_data, &[1, 3, 32, 32], &Device::Cpu).unwrap();
    let input_gpu = input_cpu.to_device(&Device::metal()).unwrap();

    let output = result
        .model
        .execute_dyn(&cache, &[&input_gpu])
        .expect("GPU execution via convert()");

    assert_eq!(output.dims(), &[1, 10]);
    assert_eq!(output.dtype(), DType::F32);

    let output_cpu = output.to_device(&Device::Cpu).unwrap();
    let vals = output_cpu.to_flat_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.is_finite(),
            "convert() dpdf GPU output[{i}] is not finite: {v}"
        );
    }

    eprintln!("[convert() dpdf backbone] output: {vals:?}");
}

/// ConvertBuilder pipeline with ConvertReport for dpdf backbone.
///
/// Validates the builder API produces a detailed report with op mapping
/// statistics, dispatch counts, and compile time for a dpdf vision model.
///
/// Part of #2293 (nn::convert() automatic model porting).
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_convert_builder_dpdf_backbone_report() {
    use crate::convert::builder::{convert as convert_build, OptLevel, VerifyLevel};

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let dir = std::env::temp_dir().join(format!("nn_convert_dpdf_rpt_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../test_data/dpdf_backbone_mini.json"),
    )
    .unwrap();
    let weights_path = write_dpdf_backbone_weights(&dir);

    let result = convert_build(&graph_path, &weights_path, &cache)
        .optimize(OptLevel::Full)
        .verify(VerifyLevel::None)
        .build()
        .expect("ConvertBuilder must succeed");

    let _ = std::fs::remove_dir_all(&dir);

    let report = &result.report;

    // Op mapping: 8 compute ops, all should be mapped.
    assert_eq!(report.op_count, 8, "8 compute nodes in dpdf backbone");
    let mapped_total: usize = report.mapped_ops.iter().map(|(_, c)| c).sum();
    assert_eq!(mapped_total, 8, "all 8 ops should be mapped");
    assert!(
        report.unmapped_ops.is_empty(),
        "no unsupported ops: {:?}",
        report.unmapped_ops
    );

    // Compilation metrics.
    assert!(report.dispatch_count >= 1, "at least 1 dispatch");
    assert!(report.total_steps >= 1, "at least 1 step");
    assert!(report.num_user_inputs == 1, "1 user input");
    assert!(report.num_weights_loaded > 0, "weights loaded");

    // Report display should not panic.
    let text = format!("{report}");
    assert!(text.contains("Conversion complete:"));

    // Report JSON should be valid.
    let json = report.to_json();
    let _val: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");

    eprintln!("[dpdf backbone report]\n{text}");
}
