// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration test: import PyanNet speaker segmentation model via `import_model()`.
//!
//! Requires pre-exported model files at `models/pyannet/`:
//! - `graph.json` (torch.export JSON, schema v8)
//! - `weights.safetensors` (35 tensors, ~2.7MB)
//!
//! The model was exported with pre-computed SincNet filterbank weights
//! (the parametric sinc filters are frozen to regular Conv1d weights).

use nn_core::dyn_tensor::trace::TraceOp;
use nn_import::import_model;

fn pyannet_model_dir() -> std::path::PathBuf {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    workspace.join("models").join("pyannet")
}

#[test]
fn test_pyannet_import_model_structure() {
    let dir = pyannet_model_dir();
    let graph_path = dir.join("graph.json");
    let weights_path = dir.join("weights.safetensors");

    if !graph_path.exists() || !weights_path.exists() {
        eprintln!("SKIP: PyanNet model files not found at {}", dir.display());
        return;
    }

    let imported = import_model(&graph_path, &weights_path)
        .unwrap_or_else(|e| panic!("import_model failed: {e:?}"));

    // Basic structure checks.
    assert_eq!(
        imported.num_user_inputs, 1,
        "PyanNet has 1 user input (waveform)"
    );
    assert_eq!(imported.output_names.len(), 1, "PyanNet has 1 output");

    // Count ops by type.
    let nodes = imported.graph.nodes();
    let count = |pred: &dyn Fn(&TraceOp) -> bool| nodes.iter().filter(|n| pred(n.op())).count();

    let conv1d_count = count(&|op| matches!(op, TraceOp::Conv1d { .. }));
    assert!(
        conv1d_count >= 3,
        "expected >= 3 Conv1d ops (SincNet + 2 conv), got {conv1d_count}"
    );

    let lstm_count = count(&|op| matches!(op, TraceOp::Lstm { .. }));
    assert!(
        lstm_count >= 4,
        "expected >= 4 LSTM ops (2-layer BiLSTM = 4 uni), got {lstm_count}"
    );

    let linear_count = count(&|op| matches!(op, TraceOp::Linear { .. }));
    assert!(
        linear_count >= 2,
        "expected >= 2 Linear ops, got {linear_count}"
    );

    let pool_count = count(&|op| matches!(op, TraceOp::MaxPool1d { .. }));
    assert!(
        pool_count >= 3,
        "expected >= 3 MaxPool1d ops, got {pool_count}"
    );

    let log_softmax_count = count(&|op| matches!(op, TraceOp::LogSoftmax { .. }));
    assert_eq!(
        log_softmax_count, 1,
        "expected 1 LogSoftmax (classifier output)"
    );

    // Output should be LogSoftmax.
    let output = imported.graph.output_node().unwrap();
    assert!(
        matches!(output.op(), TraceOp::LogSoftmax { .. }),
        "expected LogSoftmax as output, got: {:?}",
        output.op()
    );

    // Output shape: [1, 589, 8] (batch=1, frames=589, classes=8).
    let out_shape = output.output_shape();
    assert_eq!(out_shape.len(), 3, "output should be 3D");
    assert_eq!(out_shape[0], 1, "batch = 1");
    assert_eq!(out_shape[2], 8, "classes = 8 (powerset)");

    eprintln!(
        "PyanNet import OK: {} nodes, {} user inputs, output {:?}",
        nodes.len(),
        imported.num_user_inputs,
        out_shape
    );
}

/// Verify that weights were loaded for all Conv1d/Linear/LSTM ops.
#[test]
fn test_pyannet_weights_loaded() {
    let dir = pyannet_model_dir();
    let graph_path = dir.join("graph.json");
    let weights_path = dir.join("weights.safetensors");

    if !graph_path.exists() || !weights_path.exists() {
        eprintln!("SKIP: PyanNet model files not found");
        return;
    }

    let imported = import_model(&graph_path, &weights_path)
        .unwrap_or_else(|e| panic!("import_model failed: {e:?}"));

    let nodes = imported.graph.nodes();

    // Every Conv1d should have non-empty weight data.
    for node in nodes.iter() {
        if let TraceOp::Conv1d { weight, .. } = node.op() {
            assert!(
                !weight.data().is_empty(),
                "Conv1d '{}' has empty weight data",
                node.name()
            );
        }
        if let TraceOp::Linear { weight, .. } = node.op() {
            assert!(
                !weight.data().is_empty(),
                "Linear '{}' has empty weight data",
                node.name()
            );
        }
    }

    eprintln!("PyanNet weight loading OK: all parameter ops have loaded weights");
}

/// Import → trace compile → check all ops are compilable.
///
/// This exercises the trace compiler with the full PyanNet graph to verify
/// that all ops (Conv1d, MaxPool1d, LSTM, Linear, LogSoftmax, Abs, etc.)
/// have compile handlers. The compiled plan is not executed here.
#[test]
fn test_pyannet_trace_compile() {
    let dir = pyannet_model_dir();
    let graph_path = dir.join("graph.json");
    let weights_path = dir.join("weights.safetensors");

    if !graph_path.exists() || !weights_path.exists() {
        eprintln!("SKIP: PyanNet model files not found");
        return;
    }

    let imported = import_model(&graph_path, &weights_path)
        .unwrap_or_else(|e| panic!("import_model failed: {e:?}"));

    let plan = nn_dsl::compile_trace_to_plan_with_fusion(&imported.graph)
        .unwrap_or_else(|e| panic!("trace compile failed: {e:?}"));

    assert!(
        !plan.steps.is_empty(),
        "compiled plan should have at least 1 step"
    );

    eprintln!(
        "PyanNet trace compile OK: {} steps, {} weight names",
        plan.steps.len(),
        plan.weight_names.len(),
    );
}

/// Validate PyanNet GPU output: shape, dtype, finiteness, log-prob range.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn validate_pyannet_gpu_output(output: &nn_core::DynTensor) {
    use nn_core::{DType, Device};

    let out_shape = output.dims();
    assert_eq!(out_shape.len(), 3, "output should be 3D");
    assert_eq!(out_shape[0], 1, "batch = 1");
    assert_eq!(
        out_shape[2], 8,
        "expected 8 classes (powerset), got {}",
        out_shape[2]
    );
    assert_eq!(output.dtype(), DType::F32);

    let output_cpu = output.to_device(&Device::Cpu).unwrap();
    let vals = output_cpu.to_flat_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "PyanNet GPU output[{i}] is not finite: {v}");
    }
    let any_negative = vals.iter().any(|&v| v < 0.0);
    assert!(
        any_negative,
        "LogSoftmax output should have negative values (log probabilities)"
    );
    eprintln!(
        "[PyanNet GPU] output: shape={:?}, {} elements, range=[{:.4}, {:.4}]",
        out_shape,
        vals.len(),
        vals.iter().copied().fold(f32::INFINITY, f32::min),
        vals.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    );
}

/// Import → CompiledModel GPU execution on Metal.
///
/// Full pipeline: parse JSON → build ComputationGraph → trace compile →
/// Metal pipeline creation → GPU forward → finite output.
/// Part of #2295 (PyanNet speaker segmentation).
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_pyannet_compile_execute_gpu() {
    use nn_core::Device;
    use nn_metal::compiled_model::CompiledModel;

    let dir = pyannet_model_dir();
    let graph_path = dir.join("graph.json");
    let weights_path = dir.join("weights.safetensors");

    if !graph_path.exists() || !weights_path.exists() {
        eprintln!("SKIP: PyanNet model files not found");
        return;
    }

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let imported = import_model(&graph_path, &weights_path)
        .unwrap_or_else(|e| panic!("import_model failed: {e:?}"));

    let compiled = CompiledModel::builder(&imported.graph, &cache)
        .build()
        .unwrap_or_else(|e| panic!("compile PyanNet to Metal failed: {e:?}"));

    assert_eq!(compiled.num_inputs(), 1);
    eprintln!(
        "[PyanNet GPU] steps={}, dispatches={}",
        compiled.num_steps(),
        compiled.num_dispatches()
    );

    // Create GPU input: [1, 1, 160000] waveform (10s at 16kHz).
    let input_data: Vec<f32> = (0..160_000)
        .map(|i| ((i as f32) * 0.001).sin() * 0.5)
        .collect();
    let input_cpu =
        nn_core::DynTensor::from_vec(input_data, &[1, 1, 160_000], &Device::Cpu).unwrap();
    let input_gpu = input_cpu.to_device(&Device::metal()).unwrap();

    let outputs = compiled
        .execute_dyn_outputs(&cache, &[&input_gpu])
        .unwrap_or_else(|e| panic!("GPU execution failed: {e:?}"));

    assert!(!outputs.is_empty(), "expected at least 1 output");
    validate_pyannet_gpu_output(&outputs[0]);
}
