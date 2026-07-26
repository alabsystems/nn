// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! L3 parity test: compare PyanNet nn GPU output against PyTorch reference.
//!
//! Loads `models/pyannet/reference.safetensors` (18 tensors from PyTorch
//! forward pass), uses `input_0` as the reference waveform, executes the
//! compiled model on Metal GPU, and compares the `output` tensor using
//! `nn_reftest::compare_tensors()`.
//!
//! Part of #2295 (PyanNet speaker segmentation).

/// Load reference input from safetensors, execute on GPU, compare output.
///
/// Uses the exact same input waveform that PyTorch used, so any divergence
/// is purely from the nn compiled pipeline (not input differences).
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_pyannet_l3_output_parity() {
    use nn_core::{Device, DynTensor};
    use nn_import::import_model;
    use nn_metal::compiled_model::CompiledModel;
    use nn_reftest::{compare_tensors, ComparisonConfig};

    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let dir = workspace.join("models").join("pyannet");
    let graph_path = dir.join("graph.json");
    let weights_path = dir.join("weights.safetensors");
    let ref_path = dir.join("reference.safetensors");

    if !graph_path.exists() || !weights_path.exists() || !ref_path.exists() {
        eprintln!("SKIP: PyanNet model/reference files not found");
        return;
    }

    // Load reference tensors.
    let reference =
        nn_reftest::load_safetensors(&ref_path).unwrap_or_else(|e| panic!("load reference: {e}"));

    let ref_input = reference
        .get_by_name("input_0")
        .expect("reference must have 'input_0' tensor");
    let ref_output = reference
        .get_by_name("output")
        .expect("reference must have 'output' tensor");

    // Initialize Metal and compile the model.
    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let imported = import_model(&graph_path, &weights_path)
        .unwrap_or_else(|e| panic!("import_model failed: {e:?}"));
    let compiled = CompiledModel::builder(&imported.graph, &cache)
        .build()
        .unwrap_or_else(|e| panic!("compile failed: {e:?}"));

    // Use reference input (same waveform PyTorch used).
    let input_cpu =
        DynTensor::from_vec(ref_input.data.clone(), &ref_input.shape, &Device::Cpu).unwrap();
    let input_gpu = input_cpu.to_device(&Device::metal()).unwrap();

    // Execute on Metal GPU.
    let outputs = compiled
        .execute_dyn_outputs(&cache, &[&input_gpu])
        .unwrap_or_else(|e| panic!("GPU execution failed: {e:?}"));

    let output = &outputs[0];
    let output_cpu = output.to_device(&Device::Cpu).unwrap();
    let nn_vals = output_cpu.to_flat_vec::<f32>().unwrap();

    // Build candidate NamedTensor for comparison.
    let candidate =
        nn_reftest::NamedTensor::new("output", output.dims().to_vec(), nn_vals).unwrap();

    // Compare against PyTorch reference output.
    // Tolerance: abs=0.02, rel=0.02, cosine=0.999.
    // PyanNet has chained Conv1d→InstanceNorm→MaxPool1d→LSTM→Linear which
    // accumulates float32 rounding errors (~0.01 max_abs observed). Similar
    // to Whisper decoder tolerance (1e-2 abs, 5e-2 rel, 0.99 cosine).
    let config = ComparisonConfig::new(0.02, 0.02, 0.999);
    let result = compare_tensors(ref_output, &candidate, &config)
        .unwrap_or_else(|e| panic!("comparison failed: {e}"));

    eprintln!(
        "[PyanNet L3] max_abs={:.6}, mean_abs={:.6}, cosine={:.6}, rms={:.6}",
        result.max_abs_diff, result.mean_abs_diff, result.cosine_similarity, result.rms_diff
    );

    assert!(result.passed, "PyanNet L3 parity failed: {result:?}");
    eprintln!("[PyanNet L3] PASSED — nn GPU output matches PyTorch reference");
}

/// Full `convert()` pipeline for PyanNet with L3 reference parity.
///
/// This exercises the top-level `nn::convert()` API end-to-end:
/// import graph.json + weights.safetensors → compile to Metal → execute on GPU →
/// L3 parity check against reference.safetensors.
///
/// The reference trace uses PyTorch naming (`input_0`, `output`) while the graph
/// declares (`waveforms`, `log_softmax`), so this also validates the fallback
/// name resolution in `check_reference_parity`.
#[test]
#[cfg(all(feature = "metal", feature = "reftest", target_os = "macos"))]
fn test_convert_pyannet_full_pipeline_with_l3() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let dir = workspace.join("models").join("pyannet");
    let graph_path = dir.join("graph.json");
    let weights_path = dir.join("weights.safetensors");
    let ref_path = dir.join("reference.safetensors");

    if !graph_path.exists() || !weights_path.exists() || !ref_path.exists() {
        eprintln!("SKIP: PyanNet model/reference files not found");
        return;
    }

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let result = nn_import::convert(&graph_path, &weights_path, Some(&ref_path), &cache)
        .unwrap_or_else(|e| panic!("convert() failed: {e}"));

    // L3 parity: check_reference_parity should resolve names and pass.
    let parity = result
        .proof
        .reference_parity
        .as_ref()
        .expect("L3 reference parity should be populated");
    assert!(
        parity.divergence.all_passed,
        "PyanNet L3 parity via convert() must pass"
    );
    eprintln!(
        "[convert() PyanNet] L3 parity passed ({} layers checked)",
        parity.divergence.layers.len()
    );

    // Verify compiled model produces correct output shape.
    assert_eq!(result.graph.num_user_inputs, 1);
    assert_eq!(result.graph.user_input_names, vec!["waveforms"]);
    assert_eq!(result.graph.output_names, vec!["log_softmax"]);
    eprintln!("[convert() PyanNet] PASSED — full pipeline with L3 parity");
}
