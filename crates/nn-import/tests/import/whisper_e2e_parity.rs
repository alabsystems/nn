// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Whisper encoder end-to-end parity tests (#4189).
//!
//! Validates that the traced + compiled Whisper encoder produces the same
//! output as the eager forward pass, using real Whisper Tiny weights.
//!
//! # Test Architecture
//!
//! 1. **Trace+compile vs eager parity** (Metal-gated, real weights): Loads
//!    real `whisper_tiny.safetensors`, traces the encoder via `trace_graph()`,
//!    compiles to `CompiledModel`, and asserts numerical parity against eager.
//!
//! 2. **Weight loading validation** (real weights): Loads the safetensors
//!    file, verifies tensor count and key structure, and builds a WhisperModel.
//!
//! 3. **Trace graph structure** (zero weights): Traces a mini Whisper encoder
//!    and validates the computation graph structure (node types, counts).
//!
//! Part of #4189 (Whisper encoder parity test).

use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn whisper_tiny_weights_path() -> PathBuf {
    workspace_root()
        .join("models")
        .join("whisper")
        .join("whisper_tiny.safetensors")
}

// ---------------------------------------------------------------------------
// Parity helpers
// ---------------------------------------------------------------------------

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "vectors must have same length");
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = f64::from(*x);
        let y = f64::from(*y);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        if norm_a == 0.0 && norm_b == 0.0 {
            return 1.0; // Both zero vectors are "identical".
        }
        return 0.0;
    }
    dot / denom
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have same length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ---------------------------------------------------------------------------
// Test A: Trace+compile vs eager parity with real weights (Metal-gated)
// ---------------------------------------------------------------------------

/// Full end-to-end parity: load real Whisper Tiny weights, run eager encode,
/// trace the encoder, compile to CompiledModel, execute compiled, compare.
///
/// Asserts: cosine similarity > 0.9999, max absolute difference < 1e-3.
#[test]
#[cfg(target_os = "macos")]
fn test_whisper_encoder_trace_compile_parity_real_weights() {
    use nn_core::dyn_tensor::trace;
    use nn_core::dyn_tensor::DynTensor;
    use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};
    use nn_core::{DType, Device};
    use nn_metal::compiled_model::CompiledModel;
    use nn_whisper::{WhisperConfig, WhisperModel};

    let weights_path = whisper_tiny_weights_path();
    if !weights_path.exists() {
        eprintln!(
            "SKIP: Whisper Tiny weights not found at {}",
            weights_path.display()
        );
        return;
    }

    // Initialize Metal.
    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let config = WhisperConfig::whisper_tiny();
    let dev = Device::metal();

    // Load weights onto Metal device. WhisperModel::load_safetensors loads onto
    // CPU, so we load the tensors manually and move them to GPU.
    let tensors = nn_core::dyn_tensor::load_safetensors(&weights_path)
        .unwrap_or_else(|e| panic!("load safetensors: {e:?}"));

    // Move all tensors to Metal.
    let gpu_tensors: std::collections::HashMap<String, DynTensor> = tensors
        .into_iter()
        .map(|(k, v)| {
            let gpu = v
                .to_device(&dev)
                .unwrap_or_else(|e| panic!("move tensor {k} to Metal: {e:?}"));
            (k, gpu)
        })
        .collect();

    let vb = nn_core::VarBuilder::from_tensors(gpu_tensors, DType::F32, &dev);
    let mut model = WhisperModel::load(&vb, config.clone())
        .unwrap_or_else(|e| panic!("load whisper on Metal: {e:?}"));

    // Create mel input [1, 80, 3000] with small deterministic values.
    // Use a sine wave pattern to avoid all-zero degeneracy.
    let n_mel = config.num_mel_bins;
    let n_frames = 3000;
    let mel_data: Vec<f32> = (0..(n_mel * n_frames))
        .map(|i| (i as f32 * 0.001).sin() * 0.1)
        .collect();
    let mel =
        DynTensor::from_vec(mel_data, &[1, n_mel, n_frames], &dev).expect("create mel tensor");

    // Eager forward.
    let eager_output = model.encode(&mel).expect("eager encode");
    let eager_shape = eager_output.dims().to_vec();
    let eager_data = eager_output
        .to_flat_vec::<f32>()
        .expect("eager output to flat vec");

    eprintln!(
        "[Whisper Tiny real weights] eager output shape: {:?}, len={}",
        eager_shape,
        eager_data.len()
    );

    // Trace the encoder (fresh model instance to avoid cache state).
    let model_trace = WhisperModel::load(&vb, config)
        .unwrap_or_else(|e| panic!("load whisper for trace: {e:?}"));

    let (_trace_out, graph) = trace::trace_graph(|| {
        let mut traced_mel = mel.clone();
        if let Some(id) = trace::record_input(traced_mel.dims(), DType::F32) {
            traced_mel.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model_trace.encoder().forward_no_cache(&traced_mel)
        })
    })
    .expect("trace_graph for Whisper encoder");

    let node_count = graph.nodes().len();
    eprintln!(
        "[Whisper Tiny real weights] trace graph: {node_count} nodes"
    );

    // Compile to CompiledModel.
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .unwrap_or_else(|e| panic!("compile Whisper encoder: {e:?}"));

    eprintln!(
        "[Whisper Tiny real weights] compiled: {} steps, {} dispatches",
        compiled.num_steps(),
        compiled.num_dispatches()
    );

    // Execute compiled model.
    let compiled_output = compiled
        .execute_dyn(&cache, &[&mel])
        .expect("compiled execute");
    let compiled_shape = compiled_output.dims().to_vec();
    let compiled_data = compiled_output
        .to_flat_vec::<f32>()
        .expect("compiled output to flat vec");

    // Verify shapes match.
    assert_eq!(
        eager_shape, compiled_shape,
        "shape mismatch: eager={eager_shape:?} vs compiled={compiled_shape:?}"
    );

    // Numerical parity.
    let cosine = cosine_similarity(&eager_data, &compiled_data);
    let max_diff = max_abs_diff(&eager_data, &compiled_data);

    eprintln!("[Whisper Tiny real weights] cosine={cosine:.8}, max_abs_diff={max_diff:.2e}");

    assert!(
        cosine > 0.9999,
        "cosine similarity too low: {cosine:.8} (expected > 0.9999)"
    );
    assert!(
        max_diff < 1e-3,
        "max absolute difference too large: {max_diff:.2e} (expected < 1e-3)"
    );
}

// ---------------------------------------------------------------------------
// Test B: Trace graph structure validation (zero weights, always runs)
// ---------------------------------------------------------------------------

/// Trace a mini Whisper encoder with zero weights and validate graph structure.
///
/// No Metal required -- this validates the trace infrastructure produces
/// a well-formed computation graph with expected op types.
#[test]
fn test_whisper_encoder_trace_graph_structure() {
    use nn_core::dyn_tensor::trace;
    use nn_core::dyn_tensor::DynTensor;
    use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};
    use nn_core::{DType, Device, VarBuilder};
    use nn_whisper::{WhisperConfig, WhisperModel};

    let config = WhisperConfig::whisper_tiny()
        .with_num_mel_bins(4)
        .with_max_source_positions(16)
        .with_d_model(16)
        .with_encoder_attention_heads(2)
        .with_encoder_layers(2)
        .with_encoder_ffn_dim(32)
        .with_vocab_size(32)
        .with_max_target_positions(16)
        .with_decoder_attention_heads(2)
        .with_decoder_layers(1)
        .with_decoder_ffn_dim(32);

    let dev = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = WhisperModel::load(&vb, config.clone()).expect("model load");

    let n_frames = 16;
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, n_frames], DType::F32, &dev)
        .expect("mel tensor");

    let (output, graph) = trace::trace_graph(|| {
        let mut traced_mel = mel.clone();
        if let Some(id) = trace::record_input(traced_mel.dims(), DType::F32) {
            traced_mel.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model.encoder().forward_no_cache(&traced_mel)
        })
    })
    .expect("trace_graph should succeed");

    // Output shape: [1, seq_len, d_model].
    assert_eq!(output.rank(), 3);
    assert_eq!(output.dim(0).unwrap(), 1);
    assert_eq!(output.dim(2).unwrap(), config.d_model);

    // Graph should have substantial nodes.
    let node_count = graph.nodes().len();
    assert!(
        node_count > 10,
        "encoder graph should have >10 nodes, got {node_count}"
    );

    // Should have at least 1 input node.
    let input_count = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), trace::TraceOp::Input))
        .count();
    assert!(
        input_count >= 1,
        "graph should have at least 1 input, got {input_count}"
    );

    // Should have ConstantWeight nodes (conv weights, layer norm params, etc.).
    let cw_count = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), trace::TraceOp::ConstantWeight { .. }))
        .count();
    assert!(
        cw_count > 0,
        "graph should have ConstantWeight nodes, got 0"
    );

    eprintln!(
        "[Whisper mini trace] {node_count} nodes, {input_count} inputs, {cw_count} constant weights"
    );

    // Check for expected op types: at least Conv1d.
    let has_conv = graph
        .nodes()
        .iter()
        .any(|n| matches!(n.op(), trace::TraceOp::Conv1d { .. }));
    assert!(has_conv, "graph should contain Conv1d ops (encoder stem)");

    // MatMul may or may not appear depending on model size and attention path.
    // Log for diagnostics.
    let matmul_count = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), trace::TraceOp::MatMul))
        .count();
    eprintln!("[Whisper mini trace] MatMul nodes: {matmul_count}");
}

// ---------------------------------------------------------------------------
// Test C: Weight loading validation (real weights)
// ---------------------------------------------------------------------------

/// Load real Whisper Tiny weights and validate tensor structure.
///
/// Verifies: tensor count, key prefixes, and successful model construction.
#[test]
fn test_whisper_weight_loading_validation() {
    use nn_core::dyn_tensor::load_safetensors;
    use nn_core::{DType, Device, VarBuilder};
    use nn_whisper::{WhisperConfig, WhisperModel};

    let weights_path = whisper_tiny_weights_path();
    if !weights_path.exists() {
        eprintln!(
            "SKIP: Whisper Tiny weights not found at {}",
            weights_path.display()
        );
        return;
    }

    // Load raw safetensors and inspect structure.
    let tensors =
        load_safetensors(&weights_path).unwrap_or_else(|e| panic!("load safetensors: {e:?}"));

    let tensor_count = tensors.len();
    eprintln!("[Whisper weight validation] {tensor_count} tensors loaded");

    // Whisper Tiny should have a non-trivial number of tensors.
    assert!(
        tensor_count > 20,
        "expected >20 tensors for Whisper Tiny, got {tensor_count}"
    );

    // Check for expected key prefixes.
    let encoder_keys: Vec<&String> = tensors
        .keys()
        .filter(|k| k.starts_with("model.encoder"))
        .collect();
    let decoder_keys: Vec<&String> = tensors
        .keys()
        .filter(|k| k.starts_with("model.decoder"))
        .collect();

    assert!(
        !encoder_keys.is_empty(),
        "expected encoder weight keys (model.encoder.*)"
    );
    assert!(
        !decoder_keys.is_empty(),
        "expected decoder weight keys (model.decoder.*)"
    );

    eprintln!(
        "[Whisper weight validation] encoder keys: {}, decoder keys: {}",
        encoder_keys.len(),
        decoder_keys.len()
    );

    // Build WhisperModel from loaded weights to verify structural compatibility.
    let dev = Device::Cpu;
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &dev);
    let config = WhisperConfig::whisper_tiny();
    let _model = WhisperModel::load(&vb, config)
        .unwrap_or_else(|e| panic!("WhisperModel::load with real weights: {e:?}"));

    eprintln!("[Whisper weight validation] WhisperModel constructed successfully");
}
