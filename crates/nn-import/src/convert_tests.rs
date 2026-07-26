// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the convert pipeline.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;

static FIXTURE_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[test]
fn test_equivalence_proof_defaults_to_none() {
    let proof = EquivalenceProof::new(None, None, None);
    assert!(proof.kernel_safety.is_none());
    assert!(proof.composition_bounds.is_none());
    assert!(proof.reference_parity.is_none());
}

#[test]
fn test_kani_safety_report() {
    let report = KaniSafetyReport::new(538, 535, 3);
    assert_eq!(report.harness_count, 538);
    assert_eq!(report.passed, 535);
}

#[test]
fn test_composition_bounds_report() {
    let report = CompositionBoundsReport::new(true, Some(0.5));
    assert!(report.propagation_ok);
    assert!((report.output_width.unwrap() - 0.5).abs() < f32::EPSILON);
    assert!(report.composition_method.is_none());
    assert!(report.composition_soundness_mode.is_none());
    assert!(report.composition_proof_strength.is_none());
}

#[test]
fn test_composition_bounds_report_with_verifier_classification() {
    let report = CompositionBoundsReport::new(true, Some(0.5)).with_verifier_classification(
        report::ConvertCompositionMethod::Ibp,
        Some(report::ConvertSoundnessMode::Sound),
        Some(report::ConvertProofStrength::SoundIbp),
    );
    assert_eq!(
        report.composition_method,
        Some(report::ConvertCompositionMethod::Ibp)
    );
    assert_eq!(
        report.composition_soundness_mode,
        Some(report::ConvertSoundnessMode::Sound)
    );
    assert_eq!(
        report.composition_proof_strength,
        Some(report::ConvertProofStrength::SoundIbp)
    );
}

#[test]
fn test_tensor_view_to_f32_via_safetensors_roundtrip() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let raw_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "test".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 2], &raw_bytes)
            .unwrap(),
    );
    let serialized = safetensors::serialize(&tensors, None).unwrap();

    let loaded = safetensors::SafeTensors::deserialize(&serialized).unwrap();
    let view = loaded.tensor("test").unwrap();
    let result = tensor_view_to_f32(&view, "test").unwrap();
    assert_eq!(result, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_tensor_view_to_f32_u8_conversion() {
    let data: Vec<u8> = vec![0, 127, 255];
    let mut tensors = HashMap::new();
    tensors.insert(
        "u8_weights".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::U8, vec![3], &data).unwrap(),
    );
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    let loaded = safetensors::SafeTensors::deserialize(&serialized).unwrap();
    let view = loaded.tensor("u8_weights").unwrap();
    let result = tensor_view_to_f32(&view, "u8_weights").unwrap();
    assert_eq!(result, [0.0, 127.0, 255.0]);
}

#[test]
fn test_tensor_view_to_f32_i8_conversion() {
    let data: Vec<u8> = vec![0u8, 127u8, 128u8]; // 0, 127, -128 as i8
    let mut tensors = HashMap::new();
    tensors.insert(
        "i8_weights".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::I8, vec![3], &data).unwrap(),
    );
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    let loaded = safetensors::SafeTensors::deserialize(&serialized).unwrap();
    let view = loaded.tensor("i8_weights").unwrap();
    let result = tensor_view_to_f32(&view, "i8_weights").unwrap();
    assert_eq!(result, [0.0, 127.0, -128.0]);
}

#[test]
fn test_tensor_view_to_f32_unsupported_dtype_errors() {
    let data: Vec<u8> = vec![1, 0, 1];
    let mut tensors = HashMap::new();
    tensors.insert(
        "bool_mask".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::BOOL, vec![3], &data).unwrap(),
    );
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    let loaded = safetensors::SafeTensors::deserialize(&serialized).unwrap();
    let view = loaded.tensor("bool_mask").unwrap();
    let err = tensor_view_to_f32(&view, "bool_mask").unwrap_err();
    assert!(
        matches!(err, ImportError::UnsupportedDtype { .. }),
        "expected UnsupportedDtype for BOOL, got: {err:?}"
    );
}

#[test]
fn test_import_model_missing_file() {
    let err = import_model(
        Path::new("/nonexistent/graph.json"),
        Path::new("/nonexistent/weights.safetensors"),
    )
    .unwrap_err();
    assert!(
        matches!(err, ImportError::Io { .. }),
        "expected Io for missing file, got: {err:?}"
    );
}

/// Write synthetic MLP safetensors (fc1: 4->8, fc2: 8->3) to `dir`.
fn write_e2e_mlp_weights(dir: &Path) -> std::path::PathBuf {
    let fc1_w: Vec<u8> = (0..32)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    let fc1_b: Vec<u8> = [0.0f32; 8].iter().flat_map(|f| f.to_le_bytes()).collect();
    let fc2_w: Vec<u8> = (0..24)
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

/// End-to-end: write graph JSON + safetensors to disk, then call import_model().
///
/// Exercises: file I/O, JSON parse, weight load, FQN mapping, op mapping,
/// graph build, topology validation.
#[test]
fn test_import_model_end_to_end_synthetic() {
    use nn_core::dyn_tensor::trace::TraceOp;

    let dir = std::env::temp_dir().join(format!("nn_import_e2e_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let graph_path = dir.join("graph.json");
    std::fs::write(&graph_path, include_str!("../test_data/e2e_mlp.json")).unwrap();
    let weights_path = write_e2e_mlp_weights(&dir);

    let imported = import_model(&graph_path, &weights_path).unwrap();

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["linear_1"]);
    assert_eq!(imported.graph.len(), 8); // 1 input + 4 params + 3 ops

    let output = imported.graph.output_node().unwrap();
    assert!(
        matches!(output.op(), TraceOp::Linear { .. }),
        "expected Linear as output, got: {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[1, 3]);

    let linear_count = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Linear { .. }))
        .count();
    assert_eq!(linear_count, 2, "expected 2 Linear ops");

    let _ = std::fs::remove_dir_all(&dir);
}

/// bf16 weights (common in modern models).
#[test]
fn test_tensor_view_to_f32_bf16_conversion() {
    let vals = [1.0f32, -0.5, 0.25, 100.0];
    let raw: Vec<u8> = vals
        .iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
        .collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "bf16_w".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![2, 2], &raw).unwrap(),
    );
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    let loaded = safetensors::SafeTensors::deserialize(&serialized).unwrap();
    let view = loaded.tensor("bf16_w").unwrap();
    let result = tensor_view_to_f32(&view, "bf16_w").unwrap();

    for (got, expected) in result.iter().zip(&vals) {
        assert!(
            (got - expected).abs() < 0.1,
            "bf16 round-trip: got {got}, expected {expected}"
        );
    }
}

/// f16 weight conversion (used by some quantized models).
#[test]
fn test_tensor_view_to_f32_f16_conversion() {
    let vals = [0.0f32, 1.0, -1.0, 65504.0]; // max f16 value
    let raw: Vec<u8> = vals
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "f16_w".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![4], &raw).unwrap(),
    );
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    let loaded = safetensors::SafeTensors::deserialize(&serialized).unwrap();
    let view = loaded.tensor("f16_w").unwrap();
    let result = tensor_view_to_f32(&view, "f16_w").unwrap();
    assert_eq!(result.len(), 4);
    assert!((result[0] - 0.0).abs() < f32::EPSILON);
    assert!((result[1] - 1.0).abs() < f32::EPSILON);
    assert!((result[2] - (-1.0)).abs() < f32::EPSILON);
    assert!((result[3] - 65504.0).abs() < 1.0);
}

/// Write synthetic Kokoro decoder weights to a safetensors file.
///
/// Conv1: [16, 8, 3] = 384 elements, bias [16]
/// Conv2: [16, 16, 3] = 768 elements, bias [16]
fn write_kokoro_decoder_weights(dir: &Path) -> std::path::PathBuf {
    let conv1_w: Vec<u8> = (0..384)
        .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
        .collect();
    let conv1_b: Vec<u8> = [0.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
    let conv2_w: Vec<u8> = (0..768)
        .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
        .collect();
    let conv2_b: Vec<u8> = [0.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();

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

/// Import the Kokoro decoder mini fixture from disk.
fn import_kokoro_decoder_fixture() -> ImportedGraph {
    let id = FIXTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("nn_import_kokoro_{}_{}", std::process::id(), id));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../test_data/kokoro_decoder_mini.json"),
    )
    .unwrap();
    let weights_path = write_kokoro_decoder_weights(&dir);
    let imported = import_model(&graph_path, &weights_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    imported
}

/// E2E: Kokoro decoder subgraph imports with correct structure.
#[test]
fn test_import_kokoro_decoder_mini_structure() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let imported = import_kokoro_decoder_fixture();

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["output"]);
    // 1 input + 4 params + 10 compute nodes = 15
    assert_eq!(imported.graph.len(), 15);

    let output = imported.graph.output_node().unwrap();
    assert!(
        matches!(
            output.op(),
            TraceOp::Cat {
                dim: 1,
                num_inputs: 2
            }
        ),
        "expected Cat(dim=1, 2 inputs), got: {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[1, 16, 16]);
}

/// E2E: all Kokoro-specific aten ops map to correct TraceOp variants.
#[test]
fn test_import_kokoro_decoder_mini_op_counts() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let imported = import_kokoro_decoder_fixture();
    let nodes = imported.graph.nodes();
    let count = |pred: fn(&TraceOp) -> bool| nodes.iter().filter(|n| pred(n.op())).count();

    assert_eq!(count(|op| matches!(op, TraceOp::Conv1d { .. })), 2);
    assert_eq!(count(|op| matches!(op, TraceOp::InstanceNorm { .. })), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::LeakyRelu { .. })), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::Add)), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::Narrow { .. })), 2);
    assert_eq!(count(|op| matches!(op, TraceOp::Exp)), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::Sin)), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::Cat { .. })), 1);
}

/// E2E: op parameters (LeakyReLU slope, InstanceNorm eps) survive import.
#[test]
fn test_import_kokoro_decoder_mini_params() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let imported = import_kokoro_decoder_fixture();
    let nodes = imported.graph.nodes();

    let lrelu = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::LeakyRelu { .. }))
        .unwrap();
    if let TraceOp::LeakyRelu { slope } = lrelu.op() {
        assert!(
            (*slope - 0.2).abs() < 1e-6,
            "expected slope 0.2, got {slope}"
        );
    }

    let inorm = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::InstanceNorm { .. }))
        .unwrap();
    if let TraceOp::InstanceNorm { eps } = inorm.op() {
        assert!((*eps - 1e-5).abs() < 1e-8, "expected eps 1e-5, got {eps}");
    }
}

// ---------------------------------------------------------------------------
// Verification bridge tests (require `verify` feature → nn-verify + NY)
// ---------------------------------------------------------------------------

/// E2E: imported Kokoro decoder graph → NY IBP via shaped bounds.
///
/// Exercises the full pipeline: parse JSON → build ComputationGraph → translate
/// to NY GraphNetwork → IBP propagation → finite output bounds.
/// This is the L2 (composition bounds) proof layer of nn::convert().
///
/// Uses direct API with correctly shaped bounds. The `check_composition_bounds`
/// wrapper has a known bug where it creates flat 1D bounds instead of shaped
/// bounds — Conv1d-based graphs require shaped input bounds.
#[test]
#[cfg(feature = "verify")]
fn test_import_kokoro_decoder_ibp_bounds() {
    use ndarray::{ArrayD, IxDyn};

    let imported = import_kokoro_decoder_fixture();

    let gn = nn_verify::trace_to_graph_model(&imported.graph)
        .expect("trace_to_graph_model must succeed")
        .graph;

    // Use shaped bounds matching input [1, 8, 16] (not flat [128]).
    let input_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), nn_core::dyn_tensor::trace::TraceOp::Input))
        .expect("must have Input node");
    let shape = input_node.output_shape();

    let lower = ArrayD::from_elem(IxDyn(shape), -1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(shape), 1.0_f32);
    let input_bounds = nn_verify::BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    let (out_lo, out_hi) = output.lower_upper();

    // Output bounds should be finite and non-trivial.
    let max_width = out_hi
        .iter()
        .zip(out_lo.iter())
        .map(|(hi, lo)| hi - lo)
        .fold(0.0_f32, f32::max);
    assert!(
        max_width.is_finite() && max_width > 0.0,
        "output bound width should be finite and positive, got {max_width}"
    );

    eprintln!(
        "Kokoro decoder IBP: {} output elements, max_width={max_width:.4}",
        out_lo.len()
    );
}

/// Explicit trace_to_graph → IBP test with per-element bound assertions.
///
/// Unlike the above test which uses the `check_composition_bounds` wrapper,
/// this test calls `trace_to_graph_model` and `propagate_ibp` directly to
/// verify per-element bound properties on the imported Kokoro decoder.
///
/// The input BoundedTensor must match the declared input shape (NY
/// uses shaped bounds, not flat).
#[test]
#[cfg(feature = "verify")]
fn test_import_kokoro_decoder_trace_to_graph_ibp_explicit() {
    use ndarray::{ArrayD, IxDyn};

    let imported = import_kokoro_decoder_fixture();

    // Step 1: translate ComputationGraph → NY GraphNetwork.
    let gn = nn_verify::trace_to_graph_model(&imported.graph)
        .expect("trace_to_graph_model must succeed for imported Kokoro decoder")
        .graph;

    // Step 2: construct input bounds matching the declared input shape.
    // NY expects bounds shaped like the TensorSpec (including batch dim).
    // Input node shape is [1, 8, 16] from the fixture.
    let input_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), nn_core::dyn_tensor::trace::TraceOp::Input))
        .expect("imported graph must have an Input node");
    let shape = input_node.output_shape();
    assert_eq!(shape, &[1, 8, 16], "Kokoro decoder input shape");

    let lower = ArrayD::from_elem(IxDyn(shape), -1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(shape), 1.0_f32);
    let input_bounds = nn_verify::BoundedTensor::new(lower, upper).expect("valid bounds");

    // Step 3: propagate IBP bounds.
    let output = gn
        .propagate_ibp(&input_bounds)
        .expect("IBP propagation must succeed for Kokoro decoder");

    // Step 4: verify output bounds properties.
    let (out_lo, out_hi) = output.lower_upper();
    let out_size = out_lo.len();

    // All bounds must be finite (no NaN/Inf from Conv+InstanceNorm+nonlinearity chain).
    for (idx, (&lo, &hi)) in out_lo.iter().zip(out_hi.iter()).enumerate() {
        assert!(
            lo.is_finite(),
            "output lower bound at idx {idx} is not finite: {lo}"
        );
        assert!(
            hi.is_finite(),
            "output upper bound at idx {idx} is not finite: {hi}"
        );
        assert!(
            lo <= hi,
            "output bounds inverted at idx {idx}: lo={lo} > hi={hi}"
        );
    }

    // Bounds should be non-trivial (not all collapsed to zero).
    let max_width = out_hi
        .iter()
        .zip(out_lo.iter())
        .map(|(hi, lo)| hi - lo)
        .fold(0.0_f32, f32::max);
    assert!(
        max_width > 0.0,
        "IBP bounds should be non-trivial with non-zero weights, max_width={max_width}"
    );

    eprintln!("Kokoro decoder explicit IBP: {out_size} output elements, max_width={max_width:.4}",);
}

/// E2E: MLP fixture → NY IBP (simpler graph, validates Linear path).
#[test]
#[cfg(feature = "verify")]
fn test_import_mlp_ibp_bounds() {
    use ndarray::{ArrayD, IxDyn};

    let dir = std::env::temp_dir().join(format!("nn_import_mlp_ibp_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = dir.join("graph.json");
    std::fs::write(&graph_path, include_str!("../test_data/e2e_mlp.json")).unwrap();
    let weights_path = write_e2e_mlp_weights(&dir);

    let imported = import_model(&graph_path, &weights_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    let gn = nn_verify::trace_to_graph_model(&imported.graph)
        .expect("trace_to_graph_model must succeed for MLP")
        .graph;

    // MLP input shape from fixture (e.g. [1, 4]).
    let input_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), nn_core::dyn_tensor::trace::TraceOp::Input))
        .expect("must have Input node");
    let shape = input_node.output_shape();

    let lower = ArrayD::from_elem(IxDyn(shape), -1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(shape), 1.0_f32);
    let input_bounds = nn_verify::BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    let (out_lo, out_hi) = output.lower_upper();

    let max_width = out_hi
        .iter()
        .zip(out_lo.iter())
        .map(|(hi, lo)| hi - lo)
        .fold(0.0_f32, f32::max);
    assert!(
        max_width.is_finite() && max_width > 0.0,
        "MLP output bound width should be finite and positive, got {max_width}"
    );

    eprintln!(
        "MLP IBP: {} output elements, max_width={max_width:.4}",
        out_lo.len()
    );
}

// ---------------------------------------------------------------------------
// Metal GPU execution tests (require `metal` feature + macOS)
// ---------------------------------------------------------------------------

/// Import → CompiledModel::from_trace → execute_dyn on Metal GPU.
///
/// Exercises the full pipeline: parse JSON → build ComputationGraph → trace
/// compile → Metal pipeline creation → GPU forward → finite output.
/// This is the L0 (execution) proof layer of nn::convert().
///
/// Uses the MLP fixture (Linear→ReLU→Linear), the simplest multi-op graph.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_import_compile_execute_mlp_gpu() {
    use nn_core::{DType, Device};
    use nn_metal::compiled_model::CompiledModel;

    // Init Metal backend + register DynTensor GPU dispatch.
    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    // Import: JSON + safetensors → ComputationGraph.
    let dir = std::env::temp_dir().join(format!("nn_gpu_mlp_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = dir.join("graph.json");
    std::fs::write(&graph_path, include_str!("../test_data/e2e_mlp.json")).unwrap();
    let weights_path = write_e2e_mlp_weights(&dir);

    let imported = import_model(&graph_path, &weights_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    // Compile: ComputationGraph → CompiledModel (Metal pipelines).
    let compiled = CompiledModel::builder(&imported.graph, &cache)
        .build()
        .expect("compile MLP to Metal");

    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.output_shape(), &[1, 3]); // fc1: [1,4]→[1,8], fc2: [1,8]→[1,3]
    assert_eq!(compiled.output_dtype(), DType::F32);

    let nd = compiled.num_dispatches();
    assert!(nd >= 1, "expected at least 1 dispatch, got {nd}");
    eprintln!("[MLP GPU] steps={}, dispatches={nd}", compiled.num_steps());

    // Create GPU input: [1, 4] with deterministic values.
    let input_data = vec![0.5_f32, -0.3, 0.8, -0.1];
    let input_cpu = nn_core::DynTensor::from_vec(input_data, &[1, 4], &Device::Cpu).unwrap();
    let input_gpu = input_cpu.to_device(&Device::metal()).unwrap();

    // Execute on Metal GPU.
    let output = compiled
        .execute_dyn(&cache, &[&input_gpu])
        .expect("GPU execution must succeed");

    assert_eq!(output.dims(), &[1, 3]);
    assert_eq!(output.dtype(), DType::F32);

    // Move output to CPU and verify all elements are finite.
    let output_cpu = output.to_device(&Device::Cpu).unwrap();
    let vals = output_cpu.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 3);
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "MLP GPU output[{i}] is not finite: {v}");
    }

    eprintln!("[MLP GPU] output: {vals:?}");
}

/// Import → CompiledModel GPU execution for the Kokoro decoder fixture.
///
/// Tests a Conv1d-based graph (Conv→InstanceNorm→LeakyReLU→Exp→Sin→Conv→Cat)
/// which exercises the full Kokoro-style op pipeline on Metal GPU.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_import_compile_execute_kokoro_decoder_gpu() {
    use nn_core::{DType, Device};
    use nn_metal::compiled_model::CompiledModel;

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let imported = import_kokoro_decoder_fixture();

    let compiled = CompiledModel::builder(&imported.graph, &cache)
        .build()
        .expect("compile Kokoro decoder to Metal");

    assert_eq!(compiled.num_inputs(), 1);
    // Kokoro decoder output: Cat(conv1_out, conv2_narrowed) → [1, 16, 16]
    assert_eq!(compiled.output_shape(), &[1, 16, 16]);
    assert_eq!(compiled.output_dtype(), DType::F32);

    let nd = compiled.num_dispatches();
    assert!(nd >= 1, "expected at least 1 dispatch, got {nd}");
    eprintln!(
        "[Kokoro GPU] steps={}, dispatches={nd}",
        compiled.num_steps()
    );

    // Create GPU input: [1, 8, 16] matching the traced input shape.
    let input_size = 8 * 16;
    let input_data: Vec<f32> = (0..input_size).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let input_cpu = nn_core::DynTensor::from_vec(input_data, &[1, 8, 16], &Device::Cpu).unwrap();
    let input_gpu = input_cpu.to_device(&Device::metal()).unwrap();

    let output = compiled
        .execute_dyn(&cache, &[&input_gpu])
        .expect("GPU execution must succeed");

    assert_eq!(output.dims(), &[1, 16, 16]);
    assert_eq!(output.dtype(), DType::F32);

    let output_cpu = output.to_device(&Device::Cpu).unwrap();
    let vals = output_cpu.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 16 * 16);
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.is_finite(),
            "Kokoro decoder GPU output[{i}] is not finite: {v}"
        );
    }

    // Non-trivial: at least one output element should be non-zero
    // (weights are 0.001 * i, not all zero).
    let any_nonzero = vals.iter().any(|&v| v.abs() > 1e-10);
    assert!(
        any_nonzero,
        "GPU output is all zeros — likely weight loading issue"
    );

    eprintln!(
        "[Kokoro GPU] output: {} elements, range=[{:.4}, {:.4}]",
        vals.len(),
        vals.iter().copied().fold(f32::INFINITY, f32::min),
        vals.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    );
}

/// Wrapper contract: `convert()` performs the same import + compile work as the
/// explicit `import_model()` + `CompiledModel::builder()` path, while returning
/// the imported graph metadata and current report fields.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_convert_matches_manual_import_and_compile_for_exported_mlp() {
    use nn_core::{DType, Device};
    use nn_metal::compiled_model::CompiledModel;

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let id = FIXTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "nn_convert_wrapper_surface_{}_{}",
        std::process::id(),
        id
    ));
    std::fs::create_dir_all(&dir).unwrap();

    let graph_path = dir.join("graph.json");
    std::fs::write(&graph_path, include_str!("../test_data/e2e_mlp.json")).unwrap();
    let weights_path = write_e2e_mlp_weights(&dir);

    let manual_imported = import_model(&graph_path, &weights_path).expect("manual import");
    let manual_model = CompiledModel::builder(&manual_imported.graph, &cache)
        .build()
        .expect("manual compile");

    let result = convert(&graph_path, &weights_path, None, &cache).expect("convert() must succeed");

    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(
        result.graph.num_user_inputs,
        manual_imported.num_user_inputs
    );
    assert_eq!(
        result.graph.user_input_names,
        manual_imported.user_input_names
    );
    assert_eq!(result.graph.output_names, manual_imported.output_names);
    assert_eq!(result.graph.graph.len(), manual_imported.graph.len());
    assert_eq!(result.model.num_inputs(), manual_model.num_inputs());
    assert_eq!(result.model.output_shape(), manual_model.output_shape());
    assert_eq!(result.model.output_dtype(), DType::F32);
    assert!(
        result.proof.kernel_safety.is_none(),
        "Kani not run in tests"
    );
    assert!(
        result.proof.reference_parity.is_none(),
        "no reference trace provided"
    );

    let input_data = vec![0.5_f32, -0.3, 0.8, -0.1];
    let input_cpu = nn_core::DynTensor::from_vec(input_data, &[1, 4], &Device::Cpu).unwrap();
    let input_gpu = input_cpu.to_device(&Device::metal()).unwrap();

    let wrapper_output = result
        .model
        .execute_dyn(&cache, &[&input_gpu])
        .expect("wrapper GPU execution");
    let manual_output = manual_model
        .execute_dyn(&cache, &[&input_gpu])
        .expect("manual GPU execution");

    assert_eq!(wrapper_output.dims(), manual_output.dims());
    assert_eq!(wrapper_output.dtype(), DType::F32);

    let wrapper_vals = wrapper_output
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let manual_vals = manual_output
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(wrapper_vals.len(), manual_vals.len());
    for (i, (&wrapper, &manual)) in wrapper_vals.iter().zip(&manual_vals).enumerate() {
        assert!(
            (wrapper - manual).abs() < 1e-6,
            "wrapper output[{i}] diverged from manual pipeline: {wrapper} vs {manual}"
        );
    }
}

/// Full convert() pipeline: import + compile + current report scaffold.
///
/// Exercises the top-level `convert()` API which is the user-facing entry
/// point for nn::convert(). Verifies that the returned ConvertResult
/// contains a functional CompiledModel.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_convert_mlp_full_pipeline() {
    use nn_core::{DType, Device};

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let dir = std::env::temp_dir().join(format!("nn_convert_mlp_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = dir.join("graph.json");
    std::fs::write(&graph_path, include_str!("../test_data/e2e_mlp.json")).unwrap();
    let weights_path = write_e2e_mlp_weights(&dir);

    // Full convert: import + compile + current report scaffold.
    let result = convert(&graph_path, &weights_path, None, &cache).expect("convert() must succeed");

    let _ = std::fs::remove_dir_all(&dir);

    // Proof chain: L1 (Kani) is None (not run), L2 may or may not be populated.
    assert!(
        result.proof.kernel_safety.is_none(),
        "Kani not run in tests"
    );
    assert!(
        result.proof.reference_parity.is_none(),
        "no reference trace provided"
    );

    // Execute the returned model on GPU.
    let input_data = vec![0.5_f32, -0.3, 0.8, -0.1];
    let input_cpu = nn_core::DynTensor::from_vec(input_data, &[1, 4], &Device::Cpu).unwrap();
    let input_gpu = input_cpu.to_device(&Device::metal()).unwrap();

    let output = result
        .model
        .execute_dyn(&cache, &[&input_gpu])
        .expect("GPU execution via convert()");

    assert_eq!(output.dims(), &[1, 3]);
    assert_eq!(output.dtype(), DType::F32);

    let output_cpu = output.to_device(&Device::Cpu).unwrap();
    let vals = output_cpu.to_flat_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.is_finite(),
            "convert() GPU output[{i}] is not finite: {v}"
        );
    }

    eprintln!("[convert() MLP] output: {vals:?}");
}

/// Full Python→Rust integration: use files produced by `nn_export.py`
/// (graph.json + weights.safetensors + reference.safetensors) and run the
/// complete `convert()` pipeline including L3 reference parity.
///
/// This is the end-to-end test for issue #2306: export from PyTorch,
/// import in Rust, compile to Metal, execute on GPU, verify against
/// PyTorch reference activations.
#[test]
#[cfg(all(feature = "metal", feature = "reftest", target_os = "macos"))]
fn test_convert_python_exported_mlp_with_l3_parity() {
    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    // Write Python-exported fixtures to temp dir.
    let dir = std::env::temp_dir().join(format!("nn_convert_py_e2e_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../test_data/exported_mlp_graph.json"),
    )
    .unwrap();

    let weights_path = dir.join("weights.safetensors");
    std::fs::write(
        &weights_path,
        include_bytes!("../test_data/exported_mlp_weights.safetensors"),
    )
    .unwrap();

    let ref_path = dir.join("reference.safetensors");
    std::fs::write(
        &ref_path,
        include_bytes!("../test_data/exported_mlp_reference.safetensors"),
    )
    .unwrap();

    // Full convert: import + compile + proof chain with L3 parity.
    let result = convert(&graph_path, &weights_path, Some(&ref_path), &cache)
        .expect("convert() with Python-exported MLP must succeed");

    let _ = std::fs::remove_dir_all(&dir);

    // L3 parity: reference_parity must be populated and passing.
    let parity = result
        .proof
        .reference_parity
        .as_ref()
        .expect("L3 reference parity should be populated");
    assert!(
        parity.divergence.all_passed,
        "L3 parity must pass: nn GPU output matches PyTorch reference"
    );
    eprintln!(
        "[convert() Python→Rust e2e] L3 parity passed ({} layers checked)",
        parity.divergence.layers.len()
    );
}

/// Assert convert() GPU output is finite and non-trivial.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[allow(dead_code)]
fn assert_convert_gpu_output(
    result: &ConvertResult,
    cache: &nn_metal::PipelineCache,
    input_shape: &[usize],
    expected_output_shape: &[usize],
    label: &str,
) {
    use nn_core::{DType, Device};

    let numel: usize = input_shape.iter().product();
    let input_data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let input_cpu = nn_core::DynTensor::from_vec(input_data, input_shape, &Device::Cpu).unwrap();
    let input_gpu = input_cpu.to_device(&Device::metal()).unwrap();

    let output = result
        .model
        .execute_dyn(cache, &[&input_gpu])
        .unwrap_or_else(|e| panic!("[{label}] GPU execution failed: {e}"));
    assert_eq!(output.dims(), expected_output_shape);
    assert_eq!(output.dtype(), DType::F32);

    let output_cpu = output.to_device(&Device::Cpu).unwrap();
    let vals = output_cpu.to_flat_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "[{label}] output[{i}] not finite: {v}");
    }
    let any_nonzero = vals.iter().any(|&v| v.abs() > 1e-10);
    assert!(any_nonzero, "[{label}] output all zeros");
    eprintln!(
        "[{label}] output: {} elems, range=[{:.4}, {:.4}]",
        vals.len(),
        vals.iter().copied().fold(f32::INFINITY, f32::min),
        vals.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    );
}

/// Full convert() pipeline with Kokoro decoder: L0 (GPU) + L2 (IBP).
///
/// Part of #2306 (nn::convert() one-function pipeline).
#[test]
#[cfg(all(feature = "metal", feature = "verify", target_os = "macos"))]
fn test_convert_kokoro_decoder_full_pipeline() {
    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let dir = std::env::temp_dir().join(format!("nn_convert_kokoro_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../test_data/kokoro_decoder_mini.json"),
    )
    .unwrap();
    let weights_path = write_kokoro_decoder_weights(&dir);

    let result = convert(&graph_path, &weights_path, None, &cache).expect("convert() failed");
    let _ = std::fs::remove_dir_all(&dir);

    // L1/L3: absent (no Kani, no reference trace).
    assert!(result.proof.kernel_safety.is_none());
    assert!(result.proof.reference_parity.is_none());

    // L2: IBP composition bounds must propagate for Kokoro decoder.
    let l2 = result
        .proof
        .composition_bounds
        .as_ref()
        .expect("L2 composition bounds should be populated for Kokoro decoder");
    assert!(l2.propagation_ok, "IBP propagation should succeed");
    if let Some(width) = l2.output_width {
        assert!(width.is_finite() && width > 0.0, "L2 width={width}");
        eprintln!("[convert() Kokoro] L2 bound width: {width:.4}");
    }

    // L0: GPU execution.
    assert_eq!(result.graph.num_user_inputs, 1);
    assert_convert_gpu_output(
        &result,
        &cache,
        &[1, 8, 16],
        &[1, 16, 16],
        "convert() Kokoro",
    );
}

// Kokoro encoder tests extracted to convert_tests_encoder.rs (file size limit).
#[path = "convert_tests_encoder.rs"]
mod encoder;

// dpdf backbone tests extracted to convert_tests_dpdf.rs (file size limit).
#[path = "convert_tests_dpdf.rs"]
mod dpdf;

// Transformer attention tests extracted to convert_tests_attention.rs.
#[path = "convert_tests_attention.rs"]
mod attention;

// ConvBnAct backbone tests (standalone conv2d + batch_norm ops).
#[path = "convert_tests_convbn.rs"]
mod convbn;

// Normalization + Embedding tests (embedding, layer_norm, softmax, group_norm).
#[path = "convert_tests_norm_embed.rs"]
mod norm_embed;

// Pooling, reduction, and concatenation tests (MaxPool2d, AdaptiveAvgPool2d, mean, sum, cat).
#[path = "convert_tests_pool.rs"]
mod pool;

// Reshape, transpose, permute tests for multi-head attention shape ops.
#[path = "convert_tests_reshape.rs"]
mod reshape;

// Activation and elementwise op mapping tests (rsqrt, hardtanh, hardsigmoid, etc.).
#[path = "convert_tests_activation.rs"]
mod activation;

// dpdf extended op coverage tests (Wave 5: all 6 dpdf model architectures).
#[path = "convert_tests_dpdf_ops.rs"]
mod dpdf_ops;

// Wave 6: interpolate, scatter, reflection_pad2d, clamp_max op mapping tests.
#[path = "convert_tests_interp.rs"]
mod interp;
