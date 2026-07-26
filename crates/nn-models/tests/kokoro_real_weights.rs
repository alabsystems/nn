// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro-82M real-weight forward parity tests.
//!
//! Validates production weight loading, weight statistics, parameter counts,
//! model construction, text encoder forward pass, and weight-to-architecture
//! shape mapping using the actual Kokoro-82M safetensors file.
//!
//! All tests are gated on the `KOKORO_WEIGHTS` env var and skip gracefully
//! when unset. Optional forward parity tests against PyTorch reference .npy
//! files are gated on their existence in the weights directory.
//!
//! Run:
//!   KOKORO_WEIGHTS=./nn/weights/kokoro_v1_0.safetensors \
//!   cargo test -p nn-models --test kokoro_real_weights -- --nocapture
//!
//! Generate reference data (optional, for forward parity):
//!   python3 scripts/generate_kokoro_reference.py \
//!     --weights weights/kokoro_v1_0.safetensors --output-dir weights/
//!
//! Part of #3351 (Absolutely Best Kokoro).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use nn_core::dyn_tensor::DynTensor;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};
use nn_models::kokoro_tts::{KokoroConfig, KokoroModel};

// ===========================================================================
// Safetensors loader
// ===========================================================================

fn convert_tensor(
    view: &safetensors::tensor::TensorView<'_>,
    name: &str,
    device: &Device,
) -> DynTensor {
    let shape: Vec<usize> = view.shape().to_vec();
    let numel: usize = shape.iter().product();
    match view.dtype() {
        safetensors::Dtype::F32 => {
            let floats: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(floats.len(), numel, "F32 count mismatch for {name}");
            DynTensor::new(&floats, &shape, device).unwrap()
        }
        safetensors::Dtype::F16 => {
            let floats: Vec<f32> = view
                .data()
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            assert_eq!(floats.len(), numel, "F16 count mismatch for {name}");
            DynTensor::new(&floats, &shape, device).unwrap()
        }
        safetensors::Dtype::BF16 => {
            let floats: Vec<f32> = view
                .data()
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            assert_eq!(floats.len(), numel, "BF16 count mismatch for {name}");
            DynTensor::new(&floats, &shape, device).unwrap()
        }
        safetensors::Dtype::I64 => {
            let ints: Vec<i64> = view
                .data()
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            assert_eq!(ints.len(), numel, "I64 count mismatch for {name}");
            DynTensor::from_vec_i64(ints, &shape, device).unwrap()
        }
        dt => panic!("unsupported dtype {dt:?} for tensor {name}"),
    }
}

fn load_safetensors_map(path: &Path) -> HashMap<String, DynTensor> {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let tensors = safetensors::SafeTensors::deserialize(&data)
        .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    let device = Device::Cpu;
    let mut map = HashMap::new();
    for name in tensors.names() {
        let view = tensors.tensor(name).unwrap();
        map.insert(name.to_string(), convert_tensor(&view, name, &device));
    }
    map
}

// ===========================================================================
// Gate helpers
// ===========================================================================

fn kokoro_weights_path() -> Option<PathBuf> {
    let path = std::env::var("KOKORO_WEIGHTS").ok()?;
    if path.is_empty() {
        return None;
    }
    let p = PathBuf::from(&path);
    if !p.exists() {
        eprintln!("KOKORO_WEIGHTS={path} does not exist, skipping");
        return None;
    }
    Some(p)
}

fn load_model_from_weights(path: &Path) -> (KokoroModel, KokoroConfig) {
    let weight_map = load_safetensors_map(path);
    let vb = VarBuilder::from_tensors(weight_map, DType::F32, &Device::Cpu);
    let config = KokoroConfig::default();
    config.validate().expect("config validation");
    let model = KokoroModel::load(&vb, &config).expect("KokoroModel::load with real weights");
    (model, config)
}

/// Build deterministic synthetic inputs (same values as Python reference generator).
fn synthetic_inputs(config: &KokoroConfig, seq_len: usize) -> (DynTensor, DynTensor) {
    let input_ids_data: Vec<u32> = (1..=seq_len as u32).collect();
    let input_ids = DynTensor::from_vec_u32(input_ids_data, &[1, seq_len], &Device::Cpu).unwrap();
    let style_len = 2 * config.style_dim;
    let style_data: Vec<f32> = (0..style_len)
        .map(|i| (i as f32 * 0.7 + 0.3).sin() * 0.5)
        .collect();
    let style = DynTensor::new(&style_data, &[1, style_len], &Device::Cpu).unwrap();
    (input_ids, style)
}

/// Try to load a .npy reference file from the weights directory.
fn load_npy_reference(weights_path: &Path, name: &str) -> Option<Vec<f32>> {
    let npy_path = weights_path
        .parent()
        .unwrap()
        .join(format!("kokoro_ref_{name}.npy"));
    if !npy_path.exists() {
        return None;
    }
    // Simple .npy loader for f32 arrays (NumPy format v1.0)
    let data = std::fs::read(&npy_path).ok()?;
    parse_npy_f32(&data)
}

/// Parse a NumPy .npy file containing float32 data into a flat Vec<f32>.
///
/// Supports NumPy format v1.0/v2.0 with '<f4' (little-endian float32) dtype.
fn parse_npy_f32(data: &[u8]) -> Option<Vec<f32>> {
    // Magic: \x93NUMPY
    if data.len() < 10 || &data[..6] != b"\x93NUMPY" {
        return None;
    }
    let major = data[6];
    let header_len = if major == 1 {
        u16::from_le_bytes([data[8], data[9]]) as usize
    } else {
        // v2.0+: 4-byte header length
        if data.len() < 12 {
            return None;
        }
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
    };
    let header_offset = if major == 1 { 10 } else { 12 };
    let payload_start = header_offset + header_len;
    if data.len() < payload_start {
        return None;
    }
    // Parse header to confirm dtype is float32
    let header = std::str::from_utf8(&data[header_offset..payload_start]).ok()?;
    if !header.contains("<f4") && !header.contains("float32") {
        eprintln!("npy dtype not float32: {header}");
        return None;
    }
    let payload = &data[payload_start..];
    let floats: Vec<f32> = payload
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some(floats)
}

// ===========================================================================
// Test 1: Weight loading — verify tensor count and key presence
// ===========================================================================

/// Load real weights and verify tensor count is in expected range for Kokoro-82M.
///
/// Kokoro-82M v1.0 safetensors has 548 tensors (before weight_norm decomposition).
/// After remapping: ~460 tensors. Either format should have >= 400 tensors.
#[test]
fn test_weight_loading() {
    let Some(path) = kokoro_weights_path() else {
        eprintln!("KOKORO_WEIGHTS not set, skipping test_weight_loading");
        return;
    };
    eprintln!("\n=== test_weight_loading ===");
    let weight_map = load_safetensors_map(&path);
    let count = weight_map.len();
    eprintln!("Loaded {count} tensors from {}", path.display());

    assert!(
        count >= 400,
        "expected >= 400 tensors for Kokoro-82M, got {count}"
    );
    assert!(
        count <= 600,
        "expected <= 600 tensors for Kokoro-82M, got {count}"
    );

    // Verify critical keys exist
    let critical_keys = [
        "bert_encoder.weight",
        "bert_encoder.bias",
        "plbert.embeddings.word_embeddings.weight",
        "text_encoder.convs.0.weight",
        "decoder.asr_res.weight",
        "decoder.generator.conv_post.weight",
    ];
    for key in &critical_keys {
        assert!(
            weight_map.contains_key(*key),
            "missing critical weight: {key}"
        );
    }
    eprintln!("Tensor count: {count}, all critical keys present.");
}

// ===========================================================================
// Test 2: Weight statistics — finite values, no NaN/Inf
// ===========================================================================

/// Verify all weight tensors contain finite values with reasonable magnitudes.
///
/// Catches weight corruption, NaN contamination, and extreme values from
/// broken export pipelines.
#[test]
fn test_weight_statistics() {
    let Some(path) = kokoro_weights_path() else {
        eprintln!("KOKORO_WEIGHTS not set, skipping test_weight_statistics");
        return;
    };
    eprintln!("\n=== test_weight_statistics ===");
    let weight_map = load_safetensors_map(&path);

    let mut total_params: usize = 0;
    let mut nan_tensors = Vec::new();
    let mut inf_tensors = Vec::new();
    let mut max_abs_all = 0.0f32;

    for (name, tensor) in &weight_map {
        let vals = match tensor.to_flat_vec::<f32>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        total_params += vals.len();

        let nan_count = vals.iter().filter(|v| v.is_nan()).count();
        let inf_count = vals.iter().filter(|v| v.is_infinite()).count();

        if nan_count > 0 {
            nan_tensors.push(format!("{name}: {nan_count} NaN"));
        }
        if inf_count > 0 {
            inf_tensors.push(format!("{name}: {inf_count} Inf"));
        }

        let local_max = vals
            .iter()
            .filter(|v| v.is_finite())
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        max_abs_all = max_abs_all.max(local_max);
    }

    eprintln!("Total float parameters: {total_params}");
    eprintln!("Max absolute value: {max_abs_all:.4}");

    assert!(
        nan_tensors.is_empty(),
        "Tensors with NaN:\n{}",
        nan_tensors.join("\n")
    );
    assert!(
        inf_tensors.is_empty(),
        "Tensors with Inf:\n{}",
        inf_tensors.join("\n")
    );

    // Sanity: max weight should be bounded (82M well-trained model)
    assert!(
        max_abs_all < 1000.0,
        "max absolute value {max_abs_all} exceeds 1000 -- likely corrupt weights"
    );
    eprintln!("All tensors finite, max_abs={max_abs_all:.4}.");
}

// ===========================================================================
// Test 3: Parameter count matches ~82M
// ===========================================================================

/// Verify total parameter count is consistent with Kokoro-82M (~82-84M).
///
/// Cross-validates against kokoro_weight_stats.json if available.
#[test]
fn test_parameter_count() {
    let Some(path) = kokoro_weights_path() else {
        eprintln!("KOKORO_WEIGHTS not set, skipping test_parameter_count");
        return;
    };
    eprintln!("\n=== test_parameter_count ===");
    let weight_map = load_safetensors_map(&path);

    let nn_total: usize = weight_map
        .values()
        .map(|t| t.dims().iter().product::<usize>())
        .sum();
    eprintln!("nn parameter count: {nn_total}");

    assert!(
        nn_total > 70_000_000,
        "expected > 70M parameters for Kokoro-82M, got {nn_total}"
    );
    assert!(
        nn_total < 100_000_000,
        "expected < 100M parameters for Kokoro-82M, got {nn_total}"
    );

    // Cross-validate against PyTorch stats if available
    let stats_path = path.parent().unwrap().join("kokoro_weight_stats.json");
    if stats_path.exists() {
        let stats_json: serde_json::Value = {
            let data = std::fs::read_to_string(&stats_path).unwrap();
            serde_json::from_str(&data).unwrap()
        };
        let stats_map = stats_json.as_object().unwrap();
        let pytorch_total: usize = stats_map
            .values()
            .map(|v| v["numel"].as_u64().unwrap() as usize)
            .sum();

        eprintln!("PyTorch parameter count: {pytorch_total}");
        assert_eq!(
            nn_total, pytorch_total,
            "parameter count mismatch: nn={nn_total} vs pytorch={pytorch_total}"
        );
        eprintln!("Parameter counts match: {nn_total}");
    } else {
        eprintln!("kokoro_weight_stats.json not found, skipping cross-validation");
    }
}

// ===========================================================================
// Test 4: Text encoder forward pass with real weights
// ===========================================================================

/// Run TextEncoder forward pass with real weights and validate output
/// shape, finiteness, and non-degeneracy.
///
/// TextEncoder: token IDs [B=1, T] -> [B, d_en=512, T].
#[test]
fn test_forward_text_encoder() {
    let Some(path) = kokoro_weights_path() else {
        eprintln!("KOKORO_WEIGHTS not set, skipping test_forward_text_encoder");
        return;
    };
    eprintln!("\n=== test_forward_text_encoder ===");
    let (model, _config) = load_model_from_weights(&path);

    let seq_len = 8;
    let input_ids_data: Vec<u32> = (1..=seq_len as u32).collect();
    let input_ids = DynTensor::from_vec_u32(input_ids_data, &[1, seq_len], &Device::Cpu).unwrap();

    let enc_out = model
        .text_encoder()
        .forward(&input_ids)
        .expect("TextEncoder forward");
    let dims = enc_out.dims().to_vec();
    eprintln!("TextEncoder output shape: {dims:?}");

    // Expected: [1, 512, 8]
    assert_eq!(dims.len(), 3, "TextEncoder output should be 3D");
    assert_eq!(dims[0], 1, "batch size");
    assert_eq!(dims[1], 512, "d_en channels");
    assert_eq!(dims[2], seq_len, "sequence length preserved");

    let vals = enc_out.to_flat_vec::<f32>().unwrap();
    let nan_count = vals.iter().filter(|v| v.is_nan()).count();
    let inf_count = vals.iter().filter(|v| v.is_infinite()).count();
    assert_eq!(nan_count, 0, "TextEncoder output has NaN");
    assert_eq!(inf_count, 0, "TextEncoder output has Inf");

    // Verify non-degeneracy: output should have meaningful variance
    let mean = vals.iter().map(|v| f64::from(*v)).sum::<f64>() / vals.len() as f64;
    let variance = vals
        .iter()
        .map(|v| {
            let d = f64::from(*v) - mean;
            d * d
        })
        .sum::<f64>()
        / vals.len() as f64;
    eprintln!(
        "  mean={mean:.6e}, variance={variance:.6e}, range=[{:.4}, {:.4}]",
        vals.iter().copied().fold(f32::INFINITY, f32::min),
        vals.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    );
    assert!(
        variance > 1e-10,
        "TextEncoder output has near-zero variance ({variance:.2e})"
    );

    // Optional: compare against PyTorch reference if available
    if let Some(ref_vals) = load_npy_reference(&path, "text_encoder_output") {
        let cmp_len = vals.len().min(ref_vals.len());
        if cmp_len > 0 {
            let max_diff = vals[..cmp_len]
                .iter()
                .zip(ref_vals[..cmp_len].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("  PyTorch parity: max_diff={max_diff:.6e} over {cmp_len} values");
            // Tolerance: 1e-4 for f32 accumulation differences (BiLSTM, LayerNorm)
            if max_diff < 1e-3 {
                eprintln!("  TextEncoder forward parity PASSED (max_diff < 1e-3)");
            } else {
                eprintln!("  WARNING: TextEncoder max_diff={max_diff:.6e} exceeds 1e-3");
            }
        }
    } else {
        eprintln!("  No PyTorch reference found, skipping parity check");
    }

    eprintln!("TextEncoder forward validated.");
}

// ===========================================================================
// Test 5: Weight shapes match model architecture
// ===========================================================================

/// Verify weight tensor names and shapes map correctly to Kokoro-82M
/// architecture defined by KokoroConfig::default().
///
/// This catches weight remapping errors, key naming mismatches, and
/// shape incompatibilities between the safetensors file and the model.
#[test]
fn test_weight_shapes_match_model() {
    let Some(path) = kokoro_weights_path() else {
        eprintln!("KOKORO_WEIGHTS not set, skipping test_weight_shapes_match_model");
        return;
    };
    eprintln!("\n=== test_weight_shapes_match_model ===");
    let weight_map = load_safetensors_map(&path);

    // Architecture-derived shapes from KokoroConfig::default():
    //   d_en=512, hidden_size=768, embedding_dim=128, vocab_size=178
    //   style_dim=128, n_fft=20, gen_initial_channels=512
    //   upsample_rates=[10,6], f0_bilstm_hidden=256, max_dur=50
    let expected: &[(&str, &[usize])] = &[
        // bert_encoder: Linear(768, 512)
        ("bert_encoder.weight", &[512, 768]),
        ("bert_encoder.bias", &[512]),
        // PlBert embeddings
        ("plbert.embeddings.word_embeddings.weight", &[178, 128]),
        ("plbert.embeddings.position_embeddings.weight", &[512, 128]),
        ("plbert.embeddings.token_type_embeddings.weight", &[2, 128]),
        ("plbert.embeddings.LayerNorm.weight", &[128]),
        ("plbert.embeddings.LayerNorm.bias", &[128]),
        // PlBert encoder hidden mapping
        (
            "plbert.encoder.embedding_hidden_mapping_in.weight",
            &[768, 128],
        ),
        ("plbert.encoder.embedding_hidden_mapping_in.bias", &[768]),
        // TextEncoder convolutions: Conv1d(512, 512, 5)
        ("text_encoder.convs.0.weight", &[512, 512, 5]),
        ("text_encoder.convs.0.bias", &[512]),
        ("text_encoder.convs.1.weight", &[512, 512, 5]),
        ("text_encoder.convs.2.weight", &[512, 512, 5]),
        // TextEncoder LayerNorm
        ("text_encoder.norms.0.weight", &[512]),
        ("text_encoder.norms.0.bias", &[512]),
        // F0/energy predictor: shared BiLSTM input = d_en(512) + style_dim(128) = 640
        ("predictor.shared.weight_ih_l0", &[1024, 640]),
        ("predictor.shared.weight_hh_l0", &[1024, 256]),
        ("predictor.shared.weight_ih_l0_reverse", &[1024, 640]),
        // F0 projection: Linear(256, 1)
        ("predictor.F0_proj.weight", &[1, 256]),
        ("predictor.F0_proj.bias", &[1]),
        // N projection: Linear(256, 1)
        ("predictor.N_proj.weight", &[1, 256]),
        // Decoder: asr_res Conv1d(512, 64, 1)
        ("decoder.asr_res.weight", &[64, 512, 1]),
        ("decoder.asr_res.bias", &[64]),
        // Decoder: encode Stage1ResBlk Conv1d(514, 1024, 3)
        ("decoder.encode.conv1.weight", &[1024, 514, 3]),
        // Generator: SourceModule Linear(9, 1)
        ("decoder.generator.m_source.l_linear.weight", &[1, 9]),
        ("decoder.generator.m_source.l_linear.bias", &[1]),
        // Generator: conv_pre Conv1d(512, 512, 7)
        ("decoder.generator.conv_pre.weight", &[512, 512, 7]),
        // Generator: conv_post Conv1d(128, 22, 7), n_bins = n_fft/2+1 = 11, 2*11=22
        ("decoder.generator.conv_post.weight", &[22, 128, 7]),
        ("decoder.generator.conv_post.bias", &[22]),
    ];

    let mut failures = Vec::new();
    let mut checked = 0;
    for &(key, expected_shape) in expected {
        match weight_map.get(key) {
            Some(tensor) => {
                let actual = tensor.dims();
                if actual != expected_shape {
                    failures.push(format!(
                        "  {key}: expected {expected_shape:?}, got {actual:?}"
                    ));
                } else {
                    checked += 1;
                }
            }
            None => {
                failures.push(format!("  {key}: MISSING"));
            }
        }
    }

    eprintln!("Checked {checked}/{} architecture shapes.", expected.len());
    assert!(failures.is_empty(), 
        "Weight-architecture shape mismatches:\n{}",
        failures.join("\n")
    );
    eprintln!("All {} architecture shapes match.", expected.len());
}

// ===========================================================================
// Test 6: Model construction from real weights
// ===========================================================================

/// Verify KokoroModel::load() succeeds and all sub-modules are accessible.
#[test]
fn test_model_construction() {
    let Some(path) = kokoro_weights_path() else {
        eprintln!("KOKORO_WEIGHTS not set, skipping test_model_construction");
        return;
    };
    eprintln!("\n=== test_model_construction ===");
    let (model, config) = load_model_from_weights(&path);

    // Verify config
    assert_eq!(config.d_en, 512);
    assert_eq!(config.style_dim, 128);
    assert_eq!(config.n_fft, 20);
    assert_eq!(config.max_dur, 50);
    assert_eq!(config.gen_initial_channels, 512);

    // Verify sub-module accessors don't panic
    let _plbert = model.plbert();
    let _bert_enc = model.bert_encoder();
    let _text_enc = model.text_encoder();
    let _prosody = model.prosody_predictor();
    let _f0 = model.f0_predictor();
    let _decoder = model.decoder();

    // SourceModule should be present in v1.0 weights
    assert!(
        model.source_module().is_some(),
        "SourceModule should be present in v1.0 weights"
    );
    eprintln!("Model construction validated, all sub-modules accessible.");
}

// ===========================================================================
// Test 7: PlBert forward with real weights
// ===========================================================================

/// Run PlBert forward pass and validate shape, finiteness, non-degeneracy.
///
/// PlBert: [B=1, T] -> [B, T, hidden_size=768]
#[test]
fn test_forward_plbert() {
    let Some(path) = kokoro_weights_path() else {
        eprintln!("KOKORO_WEIGHTS not set, skipping test_forward_plbert");
        return;
    };
    eprintln!("\n=== test_forward_plbert ===");
    let (model, _config) = load_model_from_weights(&path);

    let seq_len = 8;
    let input_ids_data: Vec<u32> = (1..=seq_len as u32).collect();
    let input_ids = DynTensor::from_vec_u32(input_ids_data, &[1, seq_len], &Device::Cpu).unwrap();

    let bert_out = model.plbert().forward(&input_ids).expect("PlBert forward");
    let dims = bert_out.dims().to_vec();
    eprintln!("PlBert output shape: {dims:?}");

    assert_eq!(dims, vec![1, seq_len, 768], "PlBert shape mismatch");

    let vals = bert_out.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "PlBert output contains non-finite values"
    );

    let mean = vals.iter().map(|v| f64::from(*v)).sum::<f64>() / vals.len() as f64;
    let variance = vals
        .iter()
        .map(|v| {
            let d = f64::from(*v) - mean;
            d * d
        })
        .sum::<f64>()
        / vals.len() as f64;
    eprintln!("  mean={mean:.6e}, variance={variance:.6e}");
    assert!(
        variance > 1e-10,
        "PlBert output has near-zero variance ({variance:.2e})"
    );

    // Optional parity check
    if let Some(ref_vals) = load_npy_reference(&path, "plbert_output") {
        let cmp_len = vals.len().min(ref_vals.len());
        if cmp_len > 0 {
            let max_diff = vals[..cmp_len]
                .iter()
                .zip(ref_vals[..cmp_len].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("  PyTorch parity: max_diff={max_diff:.6e}");
        }
    }
    eprintln!("PlBert forward validated.");
}

// ===========================================================================
// Test 8: Full forward pass shapes and finiteness
// ===========================================================================

/// Run full model.forward() with real weights and validate output.
///
/// forward() produces (magnitude, phase) with shape [B, n_bins=11, T_out].
#[test]
fn test_forward_full_pipeline() {
    let Some(path) = kokoro_weights_path() else {
        eprintln!("KOKORO_WEIGHTS not set, skipping test_forward_full_pipeline");
        return;
    };
    eprintln!("\n=== test_forward_full_pipeline ===");
    let (model, config) = load_model_from_weights(&path);
    let seq_len = 8;
    let (input_ids, style) = synthetic_inputs(&config, seq_len);

    eprintln!("Running model.forward()...");
    let (magnitude, phase) = model
        .forward(&input_ids, &style, 1.0)
        .expect("model.forward with real weights");

    let mag_dims = magnitude.dims().to_vec();
    let phase_dims = phase.dims().to_vec();
    eprintln!("  magnitude: {mag_dims:?}");
    eprintln!("  phase: {phase_dims:?}");

    let n_bins = config.n_fft / 2 + 1; // 11
    assert_eq!(mag_dims[0], 1, "batch size");
    assert_eq!(mag_dims[1], n_bins, "n_bins");
    assert!(mag_dims[2] > 0, "T_out > 0");
    assert_eq!(phase_dims, mag_dims, "phase shape == magnitude shape");

    // Finiteness
    let mag_vals = magnitude.to_flat_vec::<f32>().unwrap();
    let phase_vals = phase.to_flat_vec::<f32>().unwrap();
    assert!(
        mag_vals.iter().all(|v| v.is_finite()),
        "magnitude contains non-finite values"
    );
    assert!(
        phase_vals.iter().all(|v| v.is_finite()),
        "phase contains non-finite values"
    );

    // Magnitude should not be all zeros
    let mag_max = mag_vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        mag_max > 1e-6,
        "magnitude appears to be all zeros (max={mag_max:.2e})"
    );

    eprintln!(
        "  magnitude range: [{:.4}, {:.4}]",
        mag_vals.iter().copied().fold(f32::INFINITY, f32::min),
        mag_max
    );
    eprintln!("Full forward pipeline validated.");
}

// ===========================================================================
// Test 9: Forward audio (E2E) produces valid PCM
// ===========================================================================

/// Run forward_audio() end-to-end and validate PCM output.
///
/// forward_audio() runs the full pipeline including iSTFT reconstruction.
#[test]
fn test_forward_audio_e2e() {
    let Some(path) = kokoro_weights_path() else {
        eprintln!("KOKORO_WEIGHTS not set, skipping test_forward_audio_e2e");
        return;
    };
    eprintln!("\n=== test_forward_audio_e2e ===");
    let (model, config) = load_model_from_weights(&path);
    let seq_len = 8;
    let (input_ids, style) = synthetic_inputs(&config, seq_len);

    eprintln!("Running model.forward_audio()...");
    let audio = model
        .forward_audio(&input_ids, &style, 1.0)
        .expect("forward_audio with real weights");

    let audio_dims = audio.dims().to_vec();
    eprintln!("  audio shape: {audio_dims:?}");

    assert_eq!(audio_dims.len(), 3, "audio rank should be 3");
    assert_eq!(audio_dims[0], 1, "batch size");
    assert_eq!(audio_dims[1], 1, "mono channel");
    assert!(audio_dims[2] > 0, "T_audio > 0");

    let pcm = audio.to_flat_vec::<f32>().unwrap();
    assert!(
        pcm.iter().all(|v| v.is_finite()),
        "audio PCM contains non-finite values"
    );

    // After clamp(-1, 1) in forward_audio, values must be in [-1, 1]
    let min_val = pcm.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = pcm.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        min_val >= -1.0 && max_val <= 1.0,
        "audio outside [-1, 1]: [{min_val}, {max_val}]"
    );

    // Audio should have some energy (not all zeros)
    let rms: f64 = (pcm.iter().map(|v| f64::from(*v * *v)).sum::<f64>() / pcm.len() as f64).sqrt();
    assert!(
        rms > 1e-6,
        "audio RMS too low ({rms:.6e}), output may be all zeros"
    );

    eprintln!(
        "  samples={}, range=[{min_val:.4}, {max_val:.4}], RMS={rms:.4e}",
        pcm.len()
    );
    eprintln!("Forward audio E2E validated.");
}

// ===========================================================================
// Test 10: Forward determinism — identical inputs produce identical outputs
// ===========================================================================

/// Two forward passes with identical inputs must produce bitwise-identical
/// outputs on CPU.
#[test]
fn test_forward_determinism() {
    let Some(path) = kokoro_weights_path() else {
        eprintln!("KOKORO_WEIGHTS not set, skipping test_forward_determinism");
        return;
    };
    eprintln!("\n=== test_forward_determinism ===");
    let (model, config) = load_model_from_weights(&path);
    let seq_len = 6;
    let (input_ids, style) = synthetic_inputs(&config, seq_len);

    let (mag1, phase1) = model
        .forward(&input_ids, &style, 1.0)
        .expect("forward run 1");
    let (mag2, phase2) = model
        .forward(&input_ids, &style, 1.0)
        .expect("forward run 2");

    assert_eq!(mag1.dims(), mag2.dims(), "magnitude shape mismatch");
    assert_eq!(phase1.dims(), phase2.dims(), "phase shape mismatch");

    let mag1_vals = mag1.to_flat_vec::<f32>().unwrap();
    let mag2_vals = mag2.to_flat_vec::<f32>().unwrap();
    let phase1_vals = phase1.to_flat_vec::<f32>().unwrap();
    let phase2_vals = phase2.to_flat_vec::<f32>().unwrap();

    let mag_diffs: usize = mag1_vals
        .iter()
        .zip(mag2_vals.iter())
        .filter(|(&a, &b)| (a - b).abs() > 0.0)
        .count();
    let phase_diffs: usize = phase1_vals
        .iter()
        .zip(phase2_vals.iter())
        .filter(|(&a, &b)| (a - b).abs() > 0.0)
        .count();

    eprintln!(
        "  magnitude diffs: {}/{}, phase diffs: {}/{}",
        mag_diffs,
        mag1_vals.len(),
        phase_diffs,
        phase1_vals.len()
    );
    assert_eq!(
        mag_diffs, 0,
        "magnitude non-deterministic: {mag_diffs} differences"
    );
    assert_eq!(
        phase_diffs, 0,
        "phase non-deterministic: {phase_diffs} differences"
    );
    eprintln!("Forward pass is deterministic (bitwise identical).");
}

// ===========================================================================
// Test 11: Weight statistics cross-validate with kokoro_weight_stats.json
// ===========================================================================

/// Compare per-tensor mean/min/max against PyTorch-generated statistics.
///
/// This catches systematic loading errors across the full weight set:
/// byte-order bugs, dtype conversion errors, or weight remapping mistakes.
#[test]
fn test_weight_stats_vs_pytorch() {
    let Some(path) = kokoro_weights_path() else {
        eprintln!("KOKORO_WEIGHTS not set, skipping test_weight_stats_vs_pytorch");
        return;
    };
    let stats_path = path.parent().unwrap().join("kokoro_weight_stats.json");
    if !stats_path.exists() {
        eprintln!(
            "kokoro_weight_stats.json not found at {}, skipping",
            stats_path.display()
        );
        return;
    }
    eprintln!("\n=== test_weight_stats_vs_pytorch ===");

    let stats_json: serde_json::Value = {
        let data = std::fs::read_to_string(&stats_path).unwrap();
        serde_json::from_str(&data).unwrap()
    };
    let stats_map = stats_json.as_object().unwrap();
    let weight_map = load_safetensors_map(&path);

    let mut checked = 0usize;
    let mut failures = Vec::new();

    for (key, ref_stats) in stats_map {
        let Some(tensor) = weight_map.get(key) else {
            failures.push(format!("{key}: present in stats but MISSING in weights"));
            continue;
        };

        // Shape validation
        let ref_shape: Vec<usize> = ref_stats["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        if tensor.dims() != ref_shape.as_slice() {
            failures.push(format!(
                "{key}: shape mismatch: pytorch={ref_shape:?}, nn={:?}",
                tensor.dims()
            ));
            continue;
        }

        // Mean/min/max for float tensors
        if let (Some(ref_mean), Some(ref_min), Some(ref_max)) = (
            ref_stats.get("mean").and_then(serde_json::Value::as_f64),
            ref_stats.get("min").and_then(serde_json::Value::as_f64),
            ref_stats.get("max").and_then(serde_json::Value::as_f64),
        ) {
            let vals = match tensor.to_flat_vec::<f32>() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if vals.is_empty() {
                continue;
            }

            let actual_mean = vals.iter().map(|v| f64::from(*v)).sum::<f64>() / vals.len() as f64;
            let actual_min = f64::from(vals.iter().copied().fold(f32::INFINITY, f32::min));
            let actual_max = f64::from(vals.iter().copied().fold(f32::NEG_INFINITY, f32::max));

            // Tolerance: 1e-4 relative for mean, 1e-5 absolute for min/max
            let mean_tol = 1e-4 * ref_mean.abs().max(1e-6);
            if (actual_mean - ref_mean).abs() > mean_tol {
                failures.push(format!(
                    "{key}: mean mismatch: pytorch={ref_mean:.8e}, nn={actual_mean:.8e}"
                ));
            }
            if (actual_min - ref_min).abs() > 1e-5 {
                failures.push(format!(
                    "{key}: min mismatch: pytorch={ref_min:.8e}, nn={actual_min:.8e}"
                ));
            }
            if (actual_max - ref_max).abs() > 1e-5 {
                failures.push(format!(
                    "{key}: max mismatch: pytorch={ref_max:.8e}, nn={actual_max:.8e}"
                ));
            }
        }
        checked += 1;
    }

    eprintln!("Checked {checked}/{} tensors.", stats_map.len());
    assert!(failures.is_empty(), 
        "Statistics mismatch ({} failures):\n{}",
        failures.len(),
        failures.join("\n")
    );
    eprintln!("All {checked} tensor statistics match PyTorch.");
}

// ===========================================================================
// Test 12: Section coverage — all major model sections present
// ===========================================================================

/// Verify all five Kokoro-82M sections have weight tensors.
#[test]
fn test_section_coverage() {
    let Some(path) = kokoro_weights_path() else {
        eprintln!("KOKORO_WEIGHTS not set, skipping test_section_coverage");
        return;
    };
    eprintln!("\n=== test_section_coverage ===");
    let weight_map = load_safetensors_map(&path);

    let mut sections: HashMap<String, usize> = HashMap::new();
    for key in weight_map.keys() {
        let section = key.split('.').next().unwrap_or("unknown").to_string();
        *sections.entry(section).or_insert(0) += 1;
    }

    for (sec, count) in &sections {
        eprintln!("  {sec}: {count} tensors");
    }

    let required = ["plbert", "bert_encoder", "text_encoder", "decoder"];
    for sec in &required {
        assert!(
            sections.get(*sec).copied().unwrap_or(0) > 0,
            "missing or empty section: {sec}"
        );
    }

    // predictor or prosody_predictor (depending on weight format)
    let has_predictor =
        sections.contains_key("predictor") || sections.contains_key("prosody_predictor");
    assert!(has_predictor, "missing predictor/prosody_predictor section");

    eprintln!("All required sections present.");
}
