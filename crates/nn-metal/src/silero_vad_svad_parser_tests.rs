// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SVAD binary parser robustness tests.
//!
//! Tests that crafted/malformed SVAD files are rejected gracefully without
//! excessive memory allocation. Part of performance_proofs phase (P1-64).

use super::super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Helper: build SVAD binary header.
fn svad_header(num_tensors: u32) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"SVAD"); // magic
    buf.extend_from_slice(&1u32.to_le_bytes()); // version 1
    buf.extend_from_slice(&num_tensors.to_le_bytes());
    buf
}

/// Helper: append a tensor entry to SVAD buffer.
fn append_tensor(buf: &mut Vec<u8>, name: &str, shape: &[u32], data: &[f32]) {
    buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
    buf.extend_from_slice(name.as_bytes());
    buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for &dim in shape {
        buf.extend_from_slice(&dim.to_le_bytes());
    }
    let data_bytes = data.len() * 4;
    buf.extend_from_slice(&(data_bytes as u32).to_le_bytes());
    for &val in data {
        buf.extend_from_slice(&val.to_le_bytes());
    }
}

/// Helper: append name bytes only to SVAD buffer.
fn append_name(buf: &mut Vec<u8>, name: &str) {
    buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
    buf.extend_from_slice(name.as_bytes());
}

/// Write bytes to a unique temp file and return the path.
fn write_temp_svad(data: &[u8]) -> std::path::PathBuf {
    let n = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("nn_svad_test_{}_{n}.svad", std::process::id()));
    std::fs::write(&path, data).expect("write temp SVAD file");
    path
}

// --- Robustness tests ---

/// SVAD with name_len exceeding the cap is rejected before allocation.
///
/// A crafted file with name_len > MAX_NAME_LEN (4096) is rejected with a
/// SvadFormat error, preventing multi-GB allocations.
#[test]
fn test_svad_excessive_name_len_rejected() {
    let mut buf = svad_header(1);
    // Inject a name_len of 1_000_000 (1 MB) — exceeds MAX_NAME_LEN (4096).
    buf.extend_from_slice(&1_000_000u32.to_le_bytes());

    let path = write_temp_svad(&buf);
    let result = SileroVadWeights::from_svad_file(&path);
    let _ = std::fs::remove_file(&path);
    let err = result.expect_err("should reject SVAD with excessive name_len");
    let err_str = format!("{err:?}");
    assert!(
        err_str.contains("name_len") && err_str.contains("exceeds maximum"),
        "expected name_len cap error, got: {err_str}",
    );
}

/// SVAD with excessive ndim is rejected before the shape-reading loop.
///
/// ndim > MAX_NDIM (16) is rejected with a SvadFormat error, preventing
/// O(ndim) CPU amplification from the shape-reading loop.
#[test]
fn test_svad_excessive_ndim_rejected() {
    let mut buf = svad_header(1);
    append_name(&mut buf, "test1");
    // ndim = 1_000_000 — exceeds MAX_NDIM (16).
    buf.extend_from_slice(&1_000_000u32.to_le_bytes());

    let path = write_temp_svad(&buf);
    let result = SileroVadWeights::from_svad_file(&path);
    let _ = std::fs::remove_file(&path);
    let err = result.expect_err("should reject SVAD with excessive ndim");
    let err_str = format!("{err:?}");
    assert!(
        err_str.contains("ndim") && err_str.contains("exceeds maximum"),
        "expected ndim cap error, got: {err_str}",
    );
}

/// SVAD with data_size exceeding remaining file is rejected before allocation.
///
/// The `data_size > cursor.len()` guard catches this before `vec![0u8; data_size]`
/// allocation, preventing OOM for crafted files where shape matches data_size
/// but the file is truncated.
#[test]
fn test_svad_excessive_data_size_rejected() {
    let mut buf = svad_header(1);
    append_name(&mut buf, "test1");
    // ndim=1, shape=[2_500_000] → shape_elems = 2_500_000
    buf.extend_from_slice(&1u32.to_le_bytes()); // ndim
    buf.extend_from_slice(&2_500_000u32.to_le_bytes()); // shape dim
                                                        // data_size = 2_500_000 * 4 = 10_000_000 (10 MB)
    buf.extend_from_slice(&10_000_000u32.to_le_bytes());
    // Provide only 100 bytes.
    buf.extend_from_slice(&[0u8; 100]);

    let path = write_temp_svad(&buf);
    let result = SileroVadWeights::from_svad_file(&path);
    let _ = std::fs::remove_file(&path);
    let err = result.expect_err("should reject SVAD with excessive data_size");
    let err_str = format!("{err:?}");
    assert!(
        err_str.contains("data_size") && err_str.contains("exceeds remaining"),
        "expected data_size exceeds remaining error, got: {err_str}",
    );
}

/// Valid minimal SVAD file parses successfully (sanity check for test helpers).
#[test]
fn test_svad_valid_minimal_round_trip() {
    let tensors: &[(&str, &[u32], usize)] = &[
        ("stft_forward_basis_buffer", &[258, 1, 256], 258 * 256),
        ("encoder_0_weight", &[128, 129, 3], 128 * 129 * 3),
        ("encoder_1_weight", &[64, 128, 3], 64 * 128 * 3),
        ("encoder_2_weight", &[64, 64, 3], 64 * 64 * 3),
        ("encoder_3_weight", &[128, 64, 3], 128 * 64 * 3),
        ("encoder_0_bias", &[128], 128),
        ("encoder_1_bias", &[64], 64),
        ("encoder_2_bias", &[64], 64),
        ("encoder_3_bias", &[128], 128),
        ("decoder_rnn_weight_ih", &[512, 128], 512 * 128),
        ("decoder_rnn_weight_hh", &[512, 128], 512 * 128),
        ("decoder_rnn_bias_ih", &[512], 512),
        ("decoder_rnn_bias_hh", &[512], 512),
        ("decoder_output_weight", &[1, 128], 128),
        ("decoder_output_bias", &[1], 1),
    ];

    let mut buf = svad_header(tensors.len() as u32);
    for &(name, shape, num_elems) in tensors {
        let data = vec![0.0f32; num_elems];
        append_tensor(&mut buf, name, shape, &data);
    }

    let path = write_temp_svad(&buf);
    let result = SileroVadWeights::from_svad_file(&path);
    let _ = std::fs::remove_file(&path);
    assert!(
        result.is_ok(),
        "valid SVAD file should parse: {:?}",
        result.err(),
    );
}

/// Bad magic is rejected.
#[test]
fn test_svad_bad_magic_rejected() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"BAAD");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());

    let path = write_temp_svad(&buf);
    let result = SileroVadWeights::from_svad_file(&path);
    let _ = std::fs::remove_file(&path);
    assert!(result.is_err());
    let err_str = format!("{:?}", result.expect_err("expected error"));
    assert!(
        err_str.contains("magic"),
        "expected bad magic error, got: {err_str}",
    );
}

/// Wrong version is rejected.
#[test]
fn test_svad_wrong_version_rejected() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"SVAD");
    buf.extend_from_slice(&99u32.to_le_bytes()); // version 99
    buf.extend_from_slice(&0u32.to_le_bytes());

    let path = write_temp_svad(&buf);
    let result = SileroVadWeights::from_svad_file(&path);
    let _ = std::fs::remove_file(&path);
    assert!(result.is_err());
    let err_str = format!("{:?}", result.expect_err("expected error"));
    assert!(
        err_str.contains("version"),
        "expected version error, got: {err_str}",
    );
}

/// Excessive num_tensors (> 1000) is rejected by the existing cap.
#[test]
fn test_svad_excessive_num_tensors_rejected() {
    let buf = svad_header(1001);
    let path = write_temp_svad(&buf);
    let result = SileroVadWeights::from_svad_file(&path);
    let _ = std::fs::remove_file(&path);
    assert!(result.is_err());
    let err_str = format!("{:?}", result.expect_err("expected error"));
    assert!(
        err_str.contains("maximum 1000"),
        "expected num_tensors cap error, got: {err_str}",
    );
}

/// Non-multiple-of-4 data_size is rejected.
#[test]
fn test_svad_data_size_not_multiple_of_4_rejected() {
    let mut buf = svad_header(1);
    append_name(&mut buf, "test1");
    buf.extend_from_slice(&1u32.to_le_bytes()); // ndim=1
    buf.extend_from_slice(&3u32.to_le_bytes()); // shape=[3] → shape_elems=3
                                                // data_size = 13 (not a multiple of 4, but also not 3*4=12)
    buf.extend_from_slice(&13u32.to_le_bytes());
    buf.extend_from_slice(&[0u8; 13]);

    let path = write_temp_svad(&buf);
    let result = SileroVadWeights::from_svad_file(&path);
    let _ = std::fs::remove_file(&path);
    assert!(result.is_err());
    let err_str = format!("{:?}", result.expect_err("expected error"));
    assert!(
        err_str.contains("multiple of 4"),
        "expected multiple-of-4 error, got: {err_str}",
    );
}

/// Shape/data_size mismatch is rejected.
#[test]
fn test_svad_shape_data_size_mismatch_rejected() {
    let mut buf = svad_header(1);
    append_name(&mut buf, "test1");
    buf.extend_from_slice(&1u32.to_le_bytes()); // ndim=1
    buf.extend_from_slice(&10u32.to_le_bytes()); // shape=[10] → shape_elems=10
                                                 // data_size = 20 (but 10 floats = 40 bytes)
    buf.extend_from_slice(&20u32.to_le_bytes());
    buf.extend_from_slice(&[0u8; 20]);

    let path = write_temp_svad(&buf);
    let result = SileroVadWeights::from_svad_file(&path);
    let _ = std::fs::remove_file(&path);
    assert!(result.is_err());
    let err_str = format!("{:?}", result.expect_err("expected error"));
    assert!(
        err_str.contains("floats") || err_str.contains("shape"),
        "expected shape/data mismatch error, got: {err_str}",
    );
}

/// AC6 (#943): SVAD file with NaN weight is rejected at load time.
///
/// A complete valid SVAD file except the STFT basis tensor contains NaN.
/// The finiteness check in `from_tensor_map()` catches this during loading.
#[test]
fn test_svad_nan_weight_rejected() {
    let tensors: &[(&str, &[u32], usize)] = &[
        ("stft_forward_basis_buffer", &[258, 1, 256], 258 * 256),
        ("encoder_0_weight", &[128, 129, 3], 128 * 129 * 3),
        ("encoder_1_weight", &[64, 128, 3], 64 * 128 * 3),
        ("encoder_2_weight", &[64, 64, 3], 64 * 64 * 3),
        ("encoder_3_weight", &[128, 64, 3], 128 * 64 * 3),
        ("encoder_0_bias", &[128], 128),
        ("encoder_1_bias", &[64], 64),
        ("encoder_2_bias", &[64], 64),
        ("encoder_3_bias", &[128], 128),
        ("decoder_rnn_weight_ih", &[512, 128], 512 * 128),
        ("decoder_rnn_weight_hh", &[512, 128], 512 * 128),
        ("decoder_rnn_bias_ih", &[512], 512),
        ("decoder_rnn_bias_hh", &[512], 512),
        ("decoder_output_weight", &[1, 128], 128),
        ("decoder_output_bias", &[1], 1),
    ];

    let mut buf = svad_header(tensors.len() as u32);
    for &(name, shape, num_elems) in tensors {
        let mut data = vec![0.5f32; num_elems];
        // Inject NaN into the STFT basis tensor.
        if name == "stft_forward_basis_buffer" {
            data[0] = f32::NAN;
            data[100] = f32::INFINITY;
        }
        append_tensor(&mut buf, name, shape, &data);
    }

    let path = write_temp_svad(&buf);
    let result = SileroVadWeights::from_svad_file(&path);
    let _ = std::fs::remove_file(&path);
    let err = result.expect_err("should reject SVAD with NaN weight");
    let err_str = format!("{err}"); // Display format: "weight tensor '...' has N non-finite value(s)"
    assert!(
        err_str.contains("non-finite") && err_str.contains("stft_forward_basis_buffer"),
        "expected NonFiniteWeight for stft_forward_basis_buffer, got: {err_str}",
    );
}

/// AC6 (#943): SVAD file with all-finite weights is still accepted.
#[test]
fn test_svad_finite_weights_accepted() {
    let tensors: &[(&str, &[u32], usize)] = &[
        ("stft_forward_basis_buffer", &[258, 1, 256], 258 * 256),
        ("encoder_0_weight", &[128, 129, 3], 128 * 129 * 3),
        ("encoder_1_weight", &[64, 128, 3], 64 * 128 * 3),
        ("encoder_2_weight", &[64, 64, 3], 64 * 64 * 3),
        ("encoder_3_weight", &[128, 64, 3], 128 * 64 * 3),
        ("encoder_0_bias", &[128], 128),
        ("encoder_1_bias", &[64], 64),
        ("encoder_2_bias", &[64], 64),
        ("encoder_3_bias", &[128], 128),
        ("decoder_rnn_weight_ih", &[512, 128], 512 * 128),
        ("decoder_rnn_weight_hh", &[512, 128], 512 * 128),
        ("decoder_rnn_bias_ih", &[512], 512),
        ("decoder_rnn_bias_hh", &[512], 512),
        ("decoder_output_weight", &[1, 128], 128),
        ("decoder_output_bias", &[1], 1),
    ];

    let mut buf = svad_header(tensors.len() as u32);
    for &(name, shape, num_elems) in tensors {
        let data: Vec<f32> = (0..num_elems).map(|i| (i as f32 * 0.001) - 0.5).collect();
        append_tensor(&mut buf, name, shape, &data);
    }

    let path = write_temp_svad(&buf);
    let result = SileroVadWeights::from_svad_file(&path);
    let _ = std::fs::remove_file(&path);
    assert!(
        result.is_ok(),
        "all-finite SVAD weights should load: {:?}",
        result.err(),
    );
}
