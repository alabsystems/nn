#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Safetensors loading tests for nn-whisper.

use crate::load_safetensors_vb;
use crate::test_utils::tiny_config;
use crate::WhisperModel;

/// Helper: write a minimal safetensors file with a single F32 tensor.
fn write_safetensors_file(path: &std::path::Path, tensors: &[(&str, &[usize], &[f32])]) {
    use std::collections::HashMap;
    let mut data_map: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();
    for (name, shape, values) in tensors {
        let bytes: &[u8] = unsafe {
            // SAFETY: f32 slice to u8 slice — same memory, valid alignment.
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4)
        };
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.to_vec(), bytes)
                .expect("valid tensor view");
        data_map.push((name.to_string(), view));
    }
    let metadata: Option<HashMap<String, String>> = None;
    let data = safetensors::tensor::serialize(
        data_map.iter().map(|(n, v)| (n.as_str(), v.clone())),
        metadata,
    )
    .expect("serialize");
    std::fs::write(path, data).expect("write file");
}

#[test]
fn test_load_safetensors_vb_reads_tensors() {
    let dir = std::env::temp_dir().join(format!("nn_whisper_st_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("model.safetensors");

    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    write_safetensors_file(&path, &[("weight", &[2, 3], &data)]);

    let vb = load_safetensors_vb(&path).expect("load vb");
    let t = vb.get(&[2, 3], "weight").expect("get tensor");
    assert_eq!(t.dims(), &[2, 3]);
    let loaded = t.to_flat_vec::<f32>().expect("flat vec");
    assert_eq!(loaded, data);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_load_safetensors_vb_rejects_nan() {
    let dir = std::env::temp_dir().join(format!("nn_whisper_st_nan_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("model.safetensors");

    let data = vec![1.0f32, f32::NAN, 3.0];
    write_safetensors_file(&path, &[("bad_weight", &[3], &data)]);

    let result = load_safetensors_vb(&path);
    assert!(result.is_err(), "should reject NaN in weight data");
    let msg = format!("{}", result.expect_err("expected error for NaN"));
    assert!(
        msg.contains("non-finite"),
        "error should mention non-finite: {msg}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_load_safetensors_vb_missing_file() {
    let result = load_safetensors_vb("/nonexistent/path/model.safetensors");
    assert!(result.is_err(), "should fail on missing file");
}

#[test]
fn test_load_safetensors_convenience() {
    // WhisperModel::load_safetensors needs real weight keys, so we verify
    // it returns an error (missing weights) rather than a parse error —
    // confirming the safetensors file is parsed successfully.
    let dir = std::env::temp_dir().join(format!("nn_whisper_st_model_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("model.safetensors");

    // Write a valid safetensors file with a dummy tensor (not Whisper weights).
    let data = vec![1.0f32; 6];
    write_safetensors_file(&path, &[("dummy", &[2, 3], &data)]);

    let config = tiny_config();
    let result = WhisperModel::load_safetensors(&path, config);
    // Should fail because Whisper weight keys are missing, not because parsing failed.
    let err = result.err().expect("should fail on missing weights");
    let msg = format!("{err}");
    assert!(
        msg.contains("not found") || msg.contains("TensorNotFound"),
        "should fail on missing weight key, got: {msg}"
    );

    std::fs::remove_dir_all(&dir).ok();
}
