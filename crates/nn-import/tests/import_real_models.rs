// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the nn-import pipeline with real model weights.
//!
//! Each test loads actual safetensors weight files and validates:
//! - All tensors parse and deserialize correctly
//! - Weight names match expected model structure
//! - Shapes and element counts are consistent
//! - Dtype conversion (F32, F16, BF16) preserves data
//! - PyTorch weight name mapping produces correct nn paths
//!
//! Tests are gated on environment variables pointing to weight files.
//! When unset, tests skip gracefully (return early).
//!
//! Required env vars:
//! - `KOKORO_WEIGHTS` — path to `kokoro_v1_0.safetensors`
//! - `SILERO_WEIGHTS` — path to `silero_vad.safetensors` (defaults to `weights/silero_vad.safetensors`)
//! - `WHISPER_WEIGHTS` — path to whisper-tiny `model.safetensors` (defaults to `weights/whisper-tiny/model.safetensors`)
//! - `GLM_WEIGHTS` — path to `glm4-tiny.safetensors` (defaults to `weights/glm4-tiny.safetensors`)
//! - `QWEN3_WEIGHTS` — path to `qwen3-0.6b.safetensors` (defaults to `weights/qwen3-0.6b.safetensors`)

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use nn_import::{
    kokoro_name_mapping, map_pytorch_key, validate_kokoro_keys, validate_kokoro_safetensors,
};

// =============================================================================
// Helpers
// =============================================================================

/// Workspace root directory (two levels up from the crate manifest).
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Resolve a weight path from an env var, with an optional default relative to workspace root.
fn resolve_weight_path(env_var: &str, default_relative: Option<&str>) -> Option<PathBuf> {
    // First check the env var.
    if let Ok(val) = std::env::var(env_var) {
        if !val.is_empty() {
            let p = PathBuf::from(&val);
            if p.exists() {
                return Some(p);
            }
        }
    }
    // Fall back to default path relative to workspace root.
    if let Some(rel) = default_relative {
        let p = workspace_root().join(rel);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Load safetensors and return (tensor_names, tensor_count, per-tensor metadata).
fn load_safetensors_metadata(
    path: &Path,
) -> (
    Vec<String>,
    usize,
    HashMap<String, (Vec<usize>, safetensors::Dtype)>,
) {
    let data = std::fs::read(path).expect("failed to read safetensors file");
    let tensors =
        safetensors::SafeTensors::deserialize(&data).expect("failed to parse safetensors");
    let names: Vec<String> = tensors.names().into_iter().map(String::from).collect();
    let count = names.len();
    let mut meta = HashMap::new();
    for name in &names {
        let view = tensors.tensor(name).unwrap();
        meta.insert(name.clone(), (view.shape().to_vec(), view.dtype()));
    }
    (names, count, meta)
}

/// Load safetensors and convert all tensors to f32, returning (name -> (f32_data, shape)).
fn load_safetensors_as_f32(path: &Path) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let data = std::fs::read(path).expect("failed to read safetensors file");
    let tensors =
        safetensors::SafeTensors::deserialize(&data).expect("failed to parse safetensors");
    let mut result = HashMap::new();
    for (name, view) in tensors.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let f32_data = tensor_view_to_f32(&view);
        result.insert(name, (f32_data, shape));
    }
    result
}

/// Convert a safetensors tensor view to f32 data (mirrors convert_weights.rs logic).
fn tensor_view_to_f32(view: &safetensors::tensor::TensorView<'_>) -> Vec<f32> {
    use safetensors::Dtype;
    let raw = view.data();
    match view.dtype() {
        Dtype::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => raw
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::BF16 => raw
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::F64 => raw
            .chunks_exact(8)
            .map(|c| {
                let bytes: [u8; 8] = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                f64::from_le_bytes(bytes) as f32
            })
            .collect(),
        Dtype::I64 => raw
            .chunks_exact(8)
            .map(|c| {
                let bytes: [u8; 8] = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                i64::from_le_bytes(bytes) as f32
            })
            .collect(),
        Dtype::U8 => raw.iter().map(|&b| f32::from(b)).collect(),
        Dtype::I8 => raw.iter().map(|&b| f32::from(b as i8)).collect(),
        other => panic!("unsupported dtype {other:?} in test helper"),
    }
}

// =============================================================================
// 1. Kokoro weight import tests
// =============================================================================

#[test]
fn test_import_kokoro_weights_structure() {
    let path = match resolve_weight_path("KOKORO_WEIGHTS", Some("weights/kokoro_v1_0.safetensors"))
    {
        Some(p) => p,
        None => {
            eprintln!("SKIP: Kokoro weights not found (set KOKORO_WEIGHTS env var)");
            return;
        }
    };

    let (names, count, meta) = load_safetensors_metadata(&path);

    // Kokoro v1.0 has 491 tensors across 6 weight groups.
    assert!(count > 400, "Kokoro should have >400 tensors, got {count}");

    // Validate all 6 required weight group prefixes are present.
    let key_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let missing = validate_kokoro_keys(&key_refs);
    assert!(
        missing.is_empty(),
        "missing Kokoro weight groups: {missing:?}"
    );

    // Validate via the full safetensors validation API.
    let mapped = validate_kokoro_safetensors(&names).expect("Kokoro validation should succeed");
    assert!(mapped > 400, "should map >400 Kokoro keys, got {mapped}");

    // Verify each tensor has consistent shape (element count matches data size).
    for (name, (shape, _dtype)) in &meta {
        let element_count: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        assert!(
            element_count > 0 || shape.is_empty(),
            "tensor '{name}' has zero elements with non-empty shape {shape:?}"
        );
    }

    // Spot-check known Kokoro tensor shapes.
    let plbert_emb = meta.get("plbert.embeddings.word_embeddings.weight");
    assert!(plbert_emb.is_some(), "plbert word embeddings must exist");
    let (emb_shape, _) = plbert_emb.unwrap();
    assert_eq!(emb_shape.len(), 2, "word embeddings should be 2D");
    assert!(
        emb_shape[0] > 100,
        "vocab size should be >100, got {}",
        emb_shape[0]
    );
}

#[test]
fn test_import_kokoro_weight_name_mapping() {
    let path = match resolve_weight_path("KOKORO_WEIGHTS", Some("weights/kokoro_v1_0.safetensors"))
    {
        Some(p) => p,
        None => {
            eprintln!("SKIP: Kokoro weights not found (set KOKORO_WEIGHTS env var)");
            return;
        }
    };

    let (names, _, _) = load_safetensors_metadata(&path);
    let mapper = kokoro_name_mapping();

    let mut unmapped = Vec::new();
    for name in &names {
        let mapped = map_pytorch_key(name);
        if mapped.is_none() {
            unmapped.push(name.as_str());
        } else {
            // Kokoro uses identity mapping -- verify mapped == original.
            assert_eq!(
                mapped.as_deref(),
                Some(name.as_str()),
                "Kokoro identity mapping broken for {name}"
            );
        }
        // Verify closure-based mapper produces the same result.
        let closure_result = mapper(name);
        assert_eq!(closure_result, *name, "closure mapper disagrees for {name}");
    }

    // Some tensors might not match the 6 known prefixes (e.g., if the model
    // format changes). Report but don't fail unless too many are unmapped.
    if !unmapped.is_empty() {
        eprintln!(
            "INFO: {}/{} Kokoro keys not in known prefixes: {:?}",
            unmapped.len(),
            names.len(),
            &unmapped[..unmapped.len().min(10)]
        );
    }
    let unmapped_ratio = unmapped.len() as f64 / names.len() as f64;
    assert!(
        unmapped_ratio < 0.1,
        "too many unmapped keys ({}/{}, {:.1}%)",
        unmapped.len(),
        names.len(),
        unmapped_ratio * 100.0
    );
}

#[test]
fn test_import_kokoro_dtype_preservation() {
    let path = match resolve_weight_path("KOKORO_WEIGHTS", Some("weights/kokoro_v1_0.safetensors"))
    {
        Some(p) => p,
        None => {
            eprintln!("SKIP: Kokoro weights not found (set KOKORO_WEIGHTS env var)");
            return;
        }
    };

    let weights = load_safetensors_as_f32(&path);

    // Verify all tensors converted to f32 without NaN/Inf corruption.
    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut total_elements = 0usize;

    for (name, (data, shape)) in &weights {
        let expected_elements: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        assert_eq!(
            data.len(),
            expected_elements,
            "tensor '{name}': data length {} != shape product {expected_elements} for shape {shape:?}",
            data.len()
        );
        total_elements += data.len();

        for &val in data {
            if val.is_nan() {
                nan_count += 1;
            }
            if val.is_infinite() {
                inf_count += 1;
            }
        }
    }

    assert!(
        total_elements > 1_000_000,
        "Kokoro should have >1M total elements, got {total_elements}"
    );
    assert_eq!(nan_count, 0, "Kokoro weights should have no NaN values");
    assert_eq!(inf_count, 0, "Kokoro weights should have no Inf values");
}

// =============================================================================
// 2. Silero VAD weight import tests
// =============================================================================

#[test]
fn test_import_silero_vad_weights() {
    let path = match resolve_weight_path("SILERO_WEIGHTS", Some("weights/silero_vad.safetensors")) {
        Some(p) => p,
        None => {
            eprintln!("SKIP: Silero VAD weights not found (set SILERO_WEIGHTS env var)");
            return;
        }
    };

    let (names, count, meta) = load_safetensors_metadata(&path);

    // Silero VAD is a compact model (~1-2 MB).
    assert!(
        count > 10,
        "Silero VAD should have >10 tensors, got {count}"
    );

    // Verify all tensors have valid shapes and dtypes.
    for (name, (shape, dtype)) in &meta {
        assert!(
            !shape.contains(&0),
            "tensor '{name}' has zero-sized dimension in shape {shape:?}"
        );
        // Silero VAD uses F32 or F16 weights.
        assert!(
            matches!(
                dtype,
                safetensors::Dtype::F32
                    | safetensors::Dtype::F16
                    | safetensors::Dtype::BF16
                    | safetensors::Dtype::I64
            ),
            "unexpected dtype {dtype:?} for tensor '{name}'"
        );
    }

    // Verify f32 conversion works for all tensors.
    let weights = load_safetensors_as_f32(&path);
    assert_eq!(weights.len(), count);

    let mut total_elements = 0usize;
    for (name, (data, shape)) in &weights {
        let expected: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        assert_eq!(
            data.len(),
            expected,
            "tensor '{name}': data length mismatch"
        );
        total_elements += data.len();

        // No NaN/Inf in weights.
        for &val in data {
            assert!(
                val.is_finite(),
                "tensor '{name}' contains non-finite value {val}"
            );
        }
    }

    assert!(
        total_elements > 1000,
        "Silero VAD should have >1000 total elements, got {total_elements}"
    );

    // Verify known Silero VAD structure: should have LSTM or encoder weights.
    let has_encoder = names
        .iter()
        .any(|n| n.contains("encoder") || n.contains("lstm"));
    let has_decoder = names
        .iter()
        .any(|n| n.contains("decoder") || n.contains("output"));
    assert!(
        has_encoder || has_decoder,
        "Silero VAD should have encoder or decoder weights, got names: {:?}",
        &names[..names.len().min(10)]
    );
}

// =============================================================================
// 3. Whisper weight import tests
// =============================================================================

#[test]
fn test_import_whisper_weights() {
    let path = match resolve_weight_path(
        "WHISPER_WEIGHTS",
        Some("weights/whisper-tiny/model.safetensors"),
    ) {
        Some(p) => p,
        None => {
            eprintln!("SKIP: Whisper weights not found (set WHISPER_WEIGHTS env var)");
            return;
        }
    };

    let (names, count, meta) = load_safetensors_metadata(&path);

    // Whisper-tiny has ~100+ tensors (encoder + decoder).
    assert!(count > 50, "Whisper should have >50 tensors, got {count}");

    // Verify encoder and decoder weight groups exist.
    let encoder_count = names.iter().filter(|n| n.contains("encoder")).count();
    let decoder_count = names.iter().filter(|n| n.contains("decoder")).count();
    assert!(encoder_count > 0, "Whisper should have encoder weights");
    assert!(decoder_count > 0, "Whisper should have decoder weights");

    // Verify known Whisper tensor structures.
    let has_embed = names.iter().any(|n| n.contains("embed"));
    assert!(has_embed, "Whisper should have embedding tensors");

    // All shapes should be valid.
    for (name, (shape, _)) in &meta {
        assert!(
            !shape.contains(&0),
            "tensor '{name}' has zero dimension: {shape:?}"
        );
    }

    // Verify f32 conversion and data integrity.
    let weights = load_safetensors_as_f32(&path);
    let mut total_elements = 0usize;
    let mut nan_count = 0usize;
    for (name, (data, shape)) in &weights {
        let expected: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        assert_eq!(
            data.len(),
            expected,
            "tensor '{name}': data length {}, expected {expected}",
            data.len()
        );
        total_elements += data.len();
        nan_count += data.iter().filter(|v| v.is_nan()).count();
    }

    assert!(
        total_elements > 100_000,
        "Whisper should have >100K elements, got {total_elements}"
    );
    assert_eq!(nan_count, 0, "Whisper weights should have no NaN values");
}

#[test]
fn test_import_whisper_attention_structure() {
    let path = match resolve_weight_path(
        "WHISPER_WEIGHTS",
        Some("weights/whisper-tiny/model.safetensors"),
    ) {
        Some(p) => p,
        None => {
            eprintln!("SKIP: Whisper weights not found (set WHISPER_WEIGHTS env var)");
            return;
        }
    };

    let (names, _, meta) = load_safetensors_metadata(&path);

    // Whisper attention layers have q_proj, k_proj, v_proj, out_proj weights.
    // Filter to `.weight` only (biases are 1D and would fail the 2D check).
    let attention_weight_keys: Vec<&String> = names
        .iter()
        .filter(|n| {
            n.contains("self_attn")
                && (n.contains("q_proj") || n.contains("k_proj"))
                && n.ends_with(".weight")
        })
        .collect();

    assert!(
        !attention_weight_keys.is_empty(),
        "Whisper should have self-attention projection weight tensors"
    );

    // For whisper-tiny, attention dimension is 384.
    // q_proj.weight shape should be [384, 384] or similar.
    for key in &attention_weight_keys {
        let (shape, _) = meta.get(*key).unwrap();
        assert_eq!(
            shape.len(),
            2,
            "attention weight '{key}' should be 2D, got shape {shape:?}"
        );
        assert!(
            shape[0] > 0 && shape[1] > 0,
            "attention weight '{key}' has invalid shape {shape:?}"
        );
    }
}

// =============================================================================
// 4. GLM-4 weight import tests
// =============================================================================

#[test]
fn test_import_glm4_weights() {
    let path = match resolve_weight_path("GLM_WEIGHTS", Some("weights/glm4-tiny.safetensors")) {
        Some(p) => p,
        None => {
            eprintln!("SKIP: GLM-4 weights not found (set GLM_WEIGHTS env var)");
            return;
        }
    };

    let (names, count, meta) = load_safetensors_metadata(&path);

    assert!(count > 10, "GLM-4 should have >10 tensors, got {count}");

    // GLM uses transformer layers with rotary embeddings.
    let has_layers = names
        .iter()
        .any(|n| n.contains("layers") || n.contains("layer"));
    assert!(has_layers, "GLM should have transformer layers");

    // Verify shapes are valid and conversion works.
    let weights = load_safetensors_as_f32(&path);
    let mut total_elements = 0usize;
    for (name, (data, shape)) in &weights {
        let expected: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        assert_eq!(
            data.len(),
            expected,
            "tensor '{name}': data {}, expected {expected}",
            data.len()
        );
        total_elements += data.len();

        // No NaN in weights.
        let nan = data.iter().filter(|v| v.is_nan()).count();
        assert_eq!(nan, 0, "tensor '{name}' has {nan} NaN values");
    }

    assert!(
        total_elements > 10_000,
        "GLM-4 should have >10K elements, got {total_elements}"
    );

    // Check for embedding table.
    let has_embed = names.iter().any(|n| n.contains("embed"));
    if has_embed {
        let embed_key = names.iter().find(|n| n.contains("embed")).unwrap();
        let (shape, _) = meta.get(embed_key).unwrap();
        assert!(
            shape.len() >= 2,
            "embedding tensor should be at least 2D, got {shape:?}"
        );
    }
}

// =============================================================================
// 5. Qwen3 weight import tests
// =============================================================================

#[test]
fn test_import_qwen3_weights() {
    let path = match resolve_weight_path("QWEN3_WEIGHTS", Some("weights/qwen3-0.6b.safetensors")) {
        Some(p) => p,
        None => {
            eprintln!("SKIP: Qwen3 weights not found (set QWEN3_WEIGHTS env var)");
            return;
        }
    };

    let (names, count, _meta) = load_safetensors_metadata(&path);

    assert!(count > 10, "Qwen3 should have >10 tensors, got {count}");

    // Qwen3 is a decoder-only transformer with SwiGLU MLP.
    let has_layers = names.iter().any(|n| n.contains("layers"));
    assert!(has_layers, "Qwen3 should have transformer layers");

    // Verify known Qwen3 MLP structure: gate_proj, up_proj, down_proj.
    let has_gate = names.iter().any(|n| n.contains("gate_proj"));
    let has_up = names.iter().any(|n| n.contains("up_proj"));
    let has_down = names.iter().any(|n| n.contains("down_proj"));
    assert!(
        has_gate && has_up && has_down,
        "Qwen3 should have SwiGLU MLP projections (gate/up/down)"
    );

    // Verify f32 conversion integrity.
    let weights = load_safetensors_as_f32(&path);
    let mut total_elements = 0usize;
    let mut nan_count = 0;
    let mut inf_count = 0;
    for (data, _) in weights.values() {
        total_elements += data.len();
        for &val in data {
            if val.is_nan() {
                nan_count += 1;
            }
            if val.is_infinite() {
                inf_count += 1;
            }
        }
    }

    assert!(
        total_elements > 100_000,
        "Qwen3-0.6B should have >100K elements, got {total_elements}"
    );
    assert_eq!(nan_count, 0, "Qwen3 weights should have no NaN");
    assert_eq!(inf_count, 0, "Qwen3 weights should have no Inf");
}

// =============================================================================
// 6. Cross-model dtype preservation tests
// =============================================================================

#[test]
fn test_dtype_preservation_across_models() {
    // Test that our f32 conversion round-trips correctly for each supported dtype.
    // Uses synthetic data to verify the conversion logic matches convert_weights.rs.

    // F32 -> F32 (identity)
    let f32_data: Vec<f32> = vec![1.0, -2.5, 0.0, 3.14, f32::MIN_POSITIVE, f32::MAX];
    let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let st = build_safetensors_typed(&[("f32", &[6], &f32_bytes, safetensors::Dtype::F32)]);
    let result = load_from_bytes(&st);
    let (data, shape) = result.get("f32").unwrap();
    assert_eq!(shape, &[6]);
    assert_eq!(data.len(), 6);
    assert!((data[0] - 1.0).abs() < f32::EPSILON);
    assert!((data[1] + 2.5).abs() < f32::EPSILON);

    // F16 -> F32
    let f16_vals = [1.0f32, -0.5, 0.0, 65504.0]; // f16 max is 65504
    let f16_bytes: Vec<u8> = f16_vals
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect();
    let st = build_safetensors_typed(&[("f16", &[4], &f16_bytes, safetensors::Dtype::F16)]);
    let result = load_from_bytes(&st);
    let (data, _) = result.get("f16").unwrap();
    assert!((data[0] - 1.0).abs() < 0.01);
    assert!((data[1] + 0.5).abs() < 0.01);
    assert_eq!(data[2], 0.0);

    // BF16 -> F32
    let bf16_vals = [1.0f32, -3.0, 0.001];
    let bf16_bytes: Vec<u8> = bf16_vals
        .iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
        .collect();
    let st = build_safetensors_typed(&[("bf16", &[3], &bf16_bytes, safetensors::Dtype::BF16)]);
    let result = load_from_bytes(&st);
    let (data, _) = result.get("bf16").unwrap();
    assert!((data[0] - 1.0).abs() < 0.02);
    assert!((data[1] + 3.0).abs() < 0.1);

    // I64 -> F32
    let i64_vals: Vec<i64> = vec![0, 1, -1, 42];
    let i64_bytes: Vec<u8> = i64_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let st = build_safetensors_typed(&[("i64", &[4], &i64_bytes, safetensors::Dtype::I64)]);
    let result = load_from_bytes(&st);
    let (data, _) = result.get("i64").unwrap();
    assert_eq!(data[0], 0.0);
    assert_eq!(data[1], 1.0);
    assert_eq!(data[2], -1.0);
    assert_eq!(data[3], 42.0);
}

/// Build safetensors from typed raw bytes.
fn build_safetensors_typed(tensors: &[(&str, &[usize], &[u8], safetensors::Dtype)]) -> Vec<u8> {
    let mut tensor_map: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();
    for &(name, shape, data, dtype) in tensors {
        let view = safetensors::tensor::TensorView::new(dtype, shape.to_vec(), data)
            .expect("valid tensor view");
        tensor_map.push((name.to_string(), view));
    }
    safetensors::tensor::serialize(tensor_map, None).expect("serialization should succeed")
}

/// Load safetensors from in-memory bytes and convert all to f32.
fn load_from_bytes(bytes: &[u8]) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let tensors =
        safetensors::SafeTensors::deserialize(bytes).expect("failed to parse safetensors");
    let mut result = HashMap::new();
    for (name, view) in tensors.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let f32_data = tensor_view_to_f32(&view);
        result.insert(name, (f32_data, shape));
    }
    result
}

// =============================================================================
// 7. Weight name mapping coverage test
// =============================================================================

#[test]
fn test_weight_name_mapping_kokoro_comprehensive() {
    // Verify that the identity mapping covers all known Kokoro weight name patterns.
    let patterns = [
        // PLBert embeddings
        "plbert.embeddings.word_embeddings.weight",
        "plbert.embeddings.position_embeddings.weight",
        "plbert.embeddings.token_type_embeddings.weight",
        "plbert.embeddings.LayerNorm.weight",
        "plbert.embeddings.LayerNorm.bias",
        // PLBert encoder
        "plbert.encoder.embedding_hidden_mapping_in.weight",
        "plbert.encoder.embedding_hidden_mapping_in.bias",
        "plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight",
        "plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight",
        "plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight",
        "plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight",
        "plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.weight",
        "plbert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight",
        "plbert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight",
        "plbert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.weight",
        // BERT encoder bridge
        "bert_encoder.weight",
        "bert_encoder.bias",
        // Text encoder LSTM
        "text_encoder.lstm.weight_ih_l0",
        "text_encoder.lstm.weight_hh_l0",
        "text_encoder.lstm.bias_ih_l0",
        "text_encoder.lstm.bias_hh_l0",
        "text_encoder.lstm.weight_ih_l0_reverse",
        "text_encoder.lstm.weight_hh_l0_reverse",
        "text_encoder.lstm.linear.weight",
        "text_encoder.lstm.linear.bias",
        // Prosody predictor
        "prosody_predictor.shared.0.conv.weight",
        "prosody_predictor.shared.0.conv.bias",
        "prosody_predictor.shared.0.lstm.weight_ih_l0",
        "prosody_predictor.norms.0.norm.weight",
        "prosody_predictor.norms.0.fc.weight",
        "prosody_predictor.duration_proj.weight",
        "prosody_predictor.duration_proj.bias",
        // Predictor
        "predictor.shared.weight_ih_l0",
        "predictor.shared.weight_ih_l0_reverse",
        "predictor.F0.0.n1.fc.weight",
        "predictor.F0.0.c1.weight",
        "predictor.F0_proj.weight",
        "predictor.N.0.n1.fc.weight",
        "predictor.N_proj.weight",
        // Decoder
        "decoder.conv_pre.weight",
        "decoder.conv_pre.bias",
        "decoder.ups.0.weight",
        "decoder.noise_convs.0.weight",
        "decoder.noise_res.0.convs1.0.weight",
        "decoder.noise_res.0.adain1.0.fc.weight",
        "decoder.noise_res.0.alpha1.0",
        "decoder.resblocks.0.convs1.0.weight",
        "decoder.resblocks.0.adain1.0.fc.weight",
        "decoder.resblocks.0.alpha1.0",
        "decoder.conv_post.weight",
        "decoder.conv_post.bias",
    ];

    for pattern in &patterns {
        let mapped = map_pytorch_key(pattern);
        assert!(
            mapped.is_some(),
            "pattern '{pattern}' should map successfully"
        );
        assert_eq!(
            mapped.as_deref(),
            Some(*pattern),
            "Kokoro identity mapping broken for '{pattern}'"
        );
    }
}

#[test]
fn test_weight_name_mapping_rejects_unknown() {
    // Ensure unknown model weight names are correctly rejected.
    let unknown_names = [
        "model.encoder.weight",
        "transformer.h.0.weight",
        "backbone.conv1.weight",
        "features.0.weight",
        "",
        "weight",
        "bias",
    ];
    for name in &unknown_names {
        assert_eq!(
            map_pytorch_key(name),
            None,
            "unknown name '{name}' should not map"
        );
    }
}

// =============================================================================
// 8. Safetensors header-only key extraction (all models)
// =============================================================================

#[test]
fn test_safetensors_key_extraction_consistency() {
    // For each available model, verify that header key extraction matches
    // full deserialization key count.
    let models: Vec<(&str, Option<PathBuf>)> = vec![
        (
            "kokoro",
            resolve_weight_path("KOKORO_WEIGHTS", Some("weights/kokoro_v1_0.safetensors")),
        ),
        (
            "silero",
            resolve_weight_path("SILERO_WEIGHTS", Some("weights/silero_vad.safetensors")),
        ),
        (
            "whisper",
            resolve_weight_path(
                "WHISPER_WEIGHTS",
                Some("weights/whisper-tiny/model.safetensors"),
            ),
        ),
        (
            "glm4",
            resolve_weight_path("GLM_WEIGHTS", Some("weights/glm4-tiny.safetensors")),
        ),
        (
            "qwen3",
            resolve_weight_path("QWEN3_WEIGHTS", Some("weights/qwen3-0.6b.safetensors")),
        ),
    ];

    let mut tested = 0;
    for (model_name, path_opt) in &models {
        let path = match path_opt {
            Some(p) => p,
            None => continue,
        };

        // Full deserialization.
        let data = std::fs::read(path).unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&data).unwrap();
        let full_names: Vec<String> = tensors.names().into_iter().map(String::from).collect();

        // Header-only extraction (same approach as kokoro_load.rs).
        let header_names = read_safetensors_keys_from_file(path);

        assert_eq!(
            full_names.len(),
            header_names.len(),
            "{model_name}: header key count ({}) != full key count ({})",
            header_names.len(),
            full_names.len()
        );

        // Verify same set of keys (order may differ).
        let mut full_sorted = full_names.clone();
        let mut header_sorted = header_names.clone();
        full_sorted.sort();
        header_sorted.sort();
        assert_eq!(
            full_sorted, header_sorted,
            "{model_name}: header keys differ from full keys"
        );

        tested += 1;
    }

    if tested == 0 {
        eprintln!("SKIP: No model weights available for key extraction test");
    }
}

/// Read safetensors keys from a file using header-only JSON parsing.
///
/// Parses the header JSON directly rather than using `read_metadata` (which
/// validates that the buffer contains the full tensor data region).
fn read_safetensors_keys_from_file(path: &Path) -> Vec<String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).unwrap();
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).unwrap();
    let header_len = u64::from_le_bytes(len_buf) as usize;

    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf).unwrap();

    // Parse the header JSON to extract tensor names.
    // The safetensors header is a JSON object mapping tensor names to metadata.
    // The special key "__metadata__" is reserved for file-level metadata.
    let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&header_buf).unwrap();
    header.into_keys().filter(|k| k != "__metadata__").collect()
}
