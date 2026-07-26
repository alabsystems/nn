// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro decoder (vocoder) — verified conversion with audio output.
//!
//! Requires pre-exported model files at `models/kokoro-82m/decoder/`:
//! - `graph.json` (torch.export JSON, ~1164 trace nodes)
//! - `weights.safetensors` (250 tensors from Generator with weight_norm removed)
//! - `reference.safetensors` (211 intermediate activations from PyTorch forward pass)
//!
//! Tests:
//! - L2: NY IBP proof — all outputs bounded for inputs in [-1, 1]
//! - L3: numerical parity with PyTorch reference activations
//! - Audio: iSTFT reconstruction → .wav file at `target/kokoro_decoder_output.wav`
//! - Full: complete `convert()` pipeline with all proof layers

// Tests are cfg-gated; helpers may appear unused without features.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn kokoro_decoder_dir() -> PathBuf {
    workspace_root()
        .join("models")
        .join("kokoro-82m")
        .join("decoder")
}

/// Returns `true` if model files exist, `false` with a skip message otherwise.
fn require_model_files(dir: &Path, files: &[&str]) -> bool {
    for f in files {
        if !dir.join(f).exists() {
            eprintln!(
                "SKIP: Kokoro decoder file '{}' not found at {}",
                f,
                dir.display()
            );
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// L2: NY IBP proof
// ---------------------------------------------------------------------------

/// Prove that the Kokoro decoder has bounded outputs for all inputs in [-1, 1].
///
/// This is the formal proof: NY's IBP propagation mathematically
/// guarantees that for ALL inputs in the domain, outputs are bounded.
/// Requires the Powf, ReflectionPad1d, and ConstantPadNd LayerSpec
/// translations added in this change.
#[test]
#[cfg(feature = "verify")]
fn test_kokoro_decoder_gamma_crown_proof() {
    let dir = kokoro_decoder_dir();
    if !require_model_files(&dir, &["graph.json", "weights.safetensors"]) {
        return;
    }

    let imported =
        nn_import::import_model(&dir.join("graph.json"), &dir.join("weights.safetensors"))
            .unwrap_or_else(|e| panic!("import_model failed: {e:?}"));

    eprintln!(
        "Kokoro decoder: {} nodes, {} user inputs",
        imported.graph.nodes().len(),
        imported.num_user_inputs
    );

    // L2: NY composition bounds.
    // The decoder has 3 inputs, so we use the multi-input translation.
    let gn = nn_verify::trace_to_graph_model_multi_input(&imported.graph)
        .unwrap_or_else(|e| panic!("trace_to_graph_model_multi_input failed: {e:?}"))
        .graph;
    eprintln!("trace_to_graph_model_multi_input succeeded — graph network built");

    // Now run the full check_composition_bounds pipeline (uses multi-input internally).
    let report = nn_import::check_composition_bounds(&imported);

    let report = report.expect("check_composition_bounds returned None — IBP propagation failed.");

    assert!(
        report.propagation_ok,
        "IBP propagation failed for Kokoro decoder"
    );

    if let Some(width) = report.output_width {
        assert!(
            width.is_finite(),
            "output bound width is non-finite: {width}"
        );
        eprintln!("Kokoro decoder L2 proof: output bound width = {width:.4}");
    }

    eprintln!("Kokoro decoder L2: NY IBP proof PASSED");
}

// ---------------------------------------------------------------------------
// L3: Reference parity
// ---------------------------------------------------------------------------

/// Confirm that the compiled model on GPU matches PyTorch reference outputs.
///
/// Uses `convert()` which loads reference.safetensors, executes the model on
/// Metal GPU with the same inputs PyTorch used, and compares outputs.
#[test]
#[cfg(all(feature = "metal", feature = "reftest", target_os = "macos"))]
fn test_kokoro_decoder_reference_parity() {
    let dir = kokoro_decoder_dir();
    if !require_model_files(
        &dir,
        &["graph.json", "weights.safetensors", "reference.safetensors"],
    ) {
        return;
    }

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let result = nn_import::convert(
        &dir.join("graph.json"),
        &dir.join("weights.safetensors"),
        Some(&dir.join("reference.safetensors")),
        &cache,
    )
    .unwrap_or_else(|e| panic!("convert() failed: {e:?}"));

    let parity = result
        .proof
        .reference_parity
        .expect("reference parity should be available");
    assert!(
        parity.divergence.all_passed,
        "L3 parity failed: {:?}",
        parity.divergence
    );

    eprintln!("Kokoro decoder L3: reference parity PASSED");
}

// ---------------------------------------------------------------------------
// Audio output
// ---------------------------------------------------------------------------

/// Execute the decoder on reference inputs, apply iSTFT, write .wav.
///
/// The decoder produces (spec, phase) tensors. Audio reconstruction:
/// `real = spec * cos(phase)`, `imag = spec * sin(phase)`, then iSTFT.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_kokoro_decoder_audio_output() {
    use nn_core::{Device, DynTensor};
    use nn_models::{IstftBasis, IstftParams};
    use std::collections::HashMap;

    let dir = kokoro_decoder_dir();
    if !require_model_files(
        &dir,
        &["graph.json", "weights.safetensors", "reference.safetensors"],
    ) {
        return;
    }

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    // Load reference BEFORE compilation so we can check for shape mismatches.
    let reference = nn_reftest::load_safetensors(dir.join("reference.safetensors"))
        .unwrap_or_else(|e| panic!("load reference: {e}"));

    // Import graph.
    let mut imported =
        nn_import::import_model(&dir.join("graph.json"), &dir.join("weights.safetensors"))
            .unwrap_or_else(|e| panic!("import_model failed: {e:?}"));

    // Build input shape overrides from reference tensors.
    // Only override input nodes — intermediate shapes are propagated.
    let mut shape_overrides: HashMap<String, Vec<usize>> = HashMap::new();

    let ref_input_keys: Vec<String> = reference
        .names()
        .filter(|k| k.starts_with("input_"))
        .map(ToString::to_string)
        .collect();
    eprintln!("Reference input keys: {ref_input_keys:?}");
    eprintln!("Graph user input names: {:?}", imported.user_input_names);

    for (i, name) in imported.user_input_names.iter().enumerate() {
        let fallback_idx = format!("input_{i}");
        let fallback_name = format!("input_{name}");
        let ref_tensor = reference
            .get_by_name(name)
            .or_else(|| reference.get_by_name(&fallback_idx))
            .or_else(|| reference.get_by_name(&fallback_name))
            .or_else(|| ref_input_keys.get(i).and_then(|k| reference.get_by_name(k)));

        if let Some(tensor) = ref_tensor {
            shape_overrides.insert(name.clone(), tensor.shape.clone());
        }
    }

    // Override input shapes and propagate through the entire graph.
    let input_updated = imported.graph.override_node_shapes(&shape_overrides);
    let propagated = imported.graph.propagate_shapes();
    let updated = input_updated + propagated;
    eprintln!(
        "Updated {updated} node shapes ({input_updated} inputs overridden, \
         {propagated} intermediates propagated, {} graph nodes total)",
        imported.graph.nodes().len()
    );

    // Compile with (potentially overridden) shapes.
    let compiled = nn_metal::compiled_model::CompiledModel::builder(&imported.graph, &cache)
        .build()
        .unwrap_or_else(|e| panic!("compile failed: {e:?}"));

    // Load reference inputs.
    let mut inputs = Vec::new();
    for (i, name) in imported.user_input_names.iter().enumerate() {
        let fallback_idx = format!("input_{i}");
        let fallback_name = format!("input_{name}");
        let tensor = reference
            .get_by_name(name)
            .or_else(|| reference.get_by_name(&fallback_idx))
            .or_else(|| reference.get_by_name(&fallback_name))
            .or_else(|| {
                // Positional fallback: i-th reference key starting with "input_".
                ref_input_keys.get(i).and_then(|k| reference.get_by_name(k))
            })
            .unwrap_or_else(|| panic!("input '{name}' not found in reference"));
        let cpu = DynTensor::from_vec(tensor.data.clone(), &tensor.shape, &Device::Cpu).unwrap();
        let gpu = cpu.to_device(&Device::metal()).unwrap();
        inputs.push(gpu);
    }

    // Execute on GPU.
    let input_refs: Vec<&DynTensor> = inputs.iter().collect();
    let outputs = compiled
        .execute_dyn_outputs(&cache, &input_refs)
        .unwrap_or_else(|e| panic!("GPU execution failed: {e:?}"));

    assert!(
        outputs.len() >= 2,
        "expected >= 2 outputs (spec, phase), got {}",
        outputs.len()
    );

    let spec_cpu = outputs[0].to_device(&Device::Cpu).unwrap();
    let phase_cpu = outputs[1].to_device(&Device::Cpu).unwrap();
    let spec_vals = spec_cpu.to_flat_vec::<f32>().unwrap();
    let phase_vals = phase_cpu.to_flat_vec::<f32>().unwrap();

    assert_eq!(
        spec_vals.len(),
        phase_vals.len(),
        "spec/phase length mismatch"
    );
    assert!(!spec_vals.is_empty(), "spec output is empty");

    eprintln!(
        "Decoder output: spec shape {:?}, phase shape {:?}",
        spec_cpu.dims(),
        phase_cpu.dims()
    );

    // Compute real/imag from spec * cos(phase), spec * sin(phase).
    let real: Vec<f32> = spec_vals
        .iter()
        .zip(phase_vals.iter())
        .map(|(&s, &p)| s * p.cos())
        .collect();
    let imag: Vec<f32> = spec_vals
        .iter()
        .zip(phase_vals.iter())
        .map(|(&s, &p)| s * p.sin())
        .collect();

    // iSTFT: Kokoro uses n_fft=20, hop=5.
    let spec_shape = spec_cpu.dims();
    let n_bins = if spec_shape.len() >= 2 {
        spec_shape[spec_shape.len() - 2]
    } else {
        spec_shape[0]
    };
    let n_frames = if spec_shape.len() >= 2 {
        spec_shape[spec_shape.len() - 1]
    } else {
        1
    };
    let n_fft = (n_bins - 1) * 2; // n_bins = n_fft/2 + 1
    let hop_length = n_fft / 4; // Kokoro default

    eprintln!(
        "iSTFT params: n_fft={n_fft}, hop={hop_length}, n_bins={n_bins}, n_frames={n_frames}"
    );

    let params = IstftParams::new(n_fft, hop_length, false, false)
        .unwrap_or_else(|e| panic!("IstftParams::new failed: {e}"));
    let basis = IstftBasis::new(params).unwrap_or_else(|e| panic!("IstftBasis::new failed: {e}"));

    let output_length = n_fft + (n_frames.saturating_sub(1)) * hop_length;
    let audio = basis
        .istft(&real, &imag, n_frames, output_length)
        .unwrap_or_else(|e| panic!("iSTFT failed: {e}"));

    assert!(!audio.is_empty(), "audio output is empty");
    assert!(
        audio.iter().all(|v| v.is_finite()),
        "audio contains non-finite values"
    );

    // RMS energy — confirm audio is not silence.
    let rms = (audio.iter().map(|v| v * v).sum::<f32>() / audio.len() as f32).sqrt();
    eprintln!("Audio: {} samples, RMS = {rms:.6}", audio.len());
    // RMS threshold is intentionally low — we just want non-silence.
    assert!(
        rms > 1e-8,
        "audio appears silent (RMS = {rms}); decoder may not be producing output"
    );

    // Write .wav file.
    let wav_path = workspace_root()
        .join("target")
        .join("kokoro_decoder_output.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&wav_path, spec)
        .unwrap_or_else(|e| panic!("wav create failed: {e}"));
    for &sample in &audio {
        writer
            .write_sample(sample)
            .unwrap_or_else(|e| panic!("wav write failed: {e}"));
    }
    writer
        .finalize()
        .unwrap_or_else(|e| panic!("wav finalize failed: {e}"));

    eprintln!(
        "Wrote {wav_path}: {} samples @ 24kHz ({:.2}s)",
        audio.len(),
        audio.len() as f64 / 24000.0,
        wav_path = wav_path.display()
    );
}

// ---------------------------------------------------------------------------
// Full convert pipeline
// ---------------------------------------------------------------------------

/// Full `convert()` pipeline with L2 + L3 proof layers.
#[test]
#[cfg(all(
    feature = "metal",
    feature = "verify",
    feature = "reftest",
    target_os = "macos"
))]
fn test_kokoro_decoder_convert_full() {
    let dir = kokoro_decoder_dir();
    if !require_model_files(
        &dir,
        &["graph.json", "weights.safetensors", "reference.safetensors"],
    ) {
        return;
    }

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let result = nn_import::convert(
        &dir.join("graph.json"),
        &dir.join("weights.safetensors"),
        Some(&dir.join("reference.safetensors")),
        &cache,
    )
    .unwrap_or_else(|e| panic!("convert() failed: {e:?}"));

    // L2: NY proof.
    let bounds = result
        .proof
        .composition_bounds
        .expect("composition bounds should be available");
    assert!(
        bounds.propagation_ok,
        "L2: IBP propagation failed for Kokoro decoder"
    );
    if let Some(width) = bounds.output_width {
        eprintln!("L2: output bound width = {width:.4}");
    }

    // L3: reference parity.
    let parity = result
        .proof
        .reference_parity
        .expect("reference parity should be available");
    assert!(
        parity.divergence.all_passed,
        "L3: parity failed: {:?}",
        parity.divergence
    );

    // Model runs: quick sanity check.
    let graph = &result.graph;
    eprintln!(
        "Full pipeline: {} nodes, {} inputs, {} outputs",
        graph.graph.nodes().len(),
        graph.num_user_inputs,
        graph.output_names.len()
    );

    eprintln!("Kokoro decoder convert full pipeline: PASSED");
}
