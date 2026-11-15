// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Regression tests: `CompiledKokoro::load` refuses a truncated or corrupted
//! weight file with an error (the mmap is validated before use) instead of
//! reading out of bounds or panicking.

use super::CompiledKokoro;

fn temp_path(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("nn_kokoro_load_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir.join(name)
}

fn valid_bytes() -> Vec<u8> {
    use safetensors::tensor::{serialize, TensorView};
    let w: Vec<f32> = vec![0.5; 16];
    let view = TensorView::new(
        safetensors::Dtype::F32,
        vec![4, 4],
        bytemuck::cast_slice(&w),
    )
    .expect("view");
    serialize(vec![("w".to_string(), view)], None).expect("serialize")
}

/// Load `bytes` from a file and return the error message; `None` if Metal is
/// unavailable on this host.
fn load_error(name: &str, bytes: &[u8]) -> Option<String> {
    if crate::metal_backend::MetalBackend::init().is_err() {
        return None;
    }
    let path = temp_path(name);
    std::fs::write(&path, bytes).expect("write");
    // SAFETY: the file is private to this test and not modified while loaded.
    let result = unsafe { CompiledKokoro::load(&path) };
    let _ = std::fs::remove_file(&path);
    match result {
        Ok(_) => panic!("{name}: expected load to fail"),
        Err(err) => Some(err.to_string()),
    }
}

#[test]
fn test_compiled_kokoro_load_truncated_file_refused() {
    let bytes = valid_bytes();
    for (label, cut) in [
        ("empty", 0),
        ("short_prefix", 5),
        ("mid_header", 20),
        ("mid_data", bytes.len() - 3),
    ] {
        let Some(msg) = load_error(&format!("trunc_{label}.safetensors"), &bytes[..cut]) else {
            return;
        };
        assert!(
            msg.contains("invalid safetensors file"),
            "truncation {label} ({cut} bytes): {msg}"
        );
    }
}

#[test]
fn test_compiled_kokoro_load_corrupted_file_refused() {
    let bytes = valid_bytes();

    let mut huge_len = bytes.clone();
    huge_len[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    let mut bad_magic = bytes.clone();
    bad_magic[8] = b'X';
    let mut header_past_end = bytes.clone();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    header_past_end[..8].copy_from_slice(&(n + 64).to_le_bytes());
    let header = r#"{"w":{"dtype":"F32","shape":[4,4],"data_offsets":[0,1048576]}}"#;
    let mut crafted_offsets = (header.len() as u64).to_le_bytes().to_vec();
    crafted_offsets.extend_from_slice(header.as_bytes());
    crafted_offsets.extend_from_slice(&[0u8; 64]);

    for (label, file) in [
        ("huge_len", huge_len),
        ("bad_magic", bad_magic),
        ("header_past_end", header_past_end),
        ("crafted_offsets", crafted_offsets),
    ] {
        let Some(msg) = load_error(&format!("corrupt_{label}.safetensors"), &file) else {
            return;
        };
        assert!(msg.contains("invalid safetensors file"), "{label}: {msg}");
    }
}

/// A structurally valid file passes validation and fails later, on the
/// missing Kokoro weights — so validation does not refuse good files.
#[test]
fn test_compiled_kokoro_load_valid_file_passes_validation() {
    let Some(msg) = load_error("valid_but_not_kokoro.safetensors", &valid_bytes()) else {
        return;
    };
    assert!(!msg.contains("invalid safetensors file"), "{msg}");
}
