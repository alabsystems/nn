// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! External tests for `spirv_reduction` module.
//!
//! Tests the public API from outside the module, exercising SPIR-V generation
//! for sum, max, mean, and softmax reduction kernels.

use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};
use crate::spirv_emit::SPIRV_MAGIC;
use crate::spirv_reduction::{
    generate_max_spirv, generate_mean_spirv, generate_softmax_spirv, generate_sum_spirv,
    REDUCTION_WORKGROUP_SIZE,
};

/// Convert a SPIR-V byte array to a word array for header inspection.
fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0, "SPIR-V binary must be 4-byte aligned");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Assert SPIR-V magic number at the start of the byte stream.
fn assert_spirv_magic_bytes(bytes: &[u8], label: &str) {
    assert!(bytes.len() >= 4, "{label}: module too short for magic");
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(magic, SPIRV_MAGIC, "{label}: wrong SPIR-V magic number");
}

// ---- test_sum_spirv_valid ----

#[test]
fn test_sum_spirv_valid() {
    let bytes = generate_sum_spirv(1024);
    assert_spirv_magic_bytes(&bytes, "sum_1024");
    assert_eq!(bytes.len() % 4, 0, "sum must be 4-byte aligned");
    let words = bytes_to_words(&bytes);
    assert!(words.len() >= 5, "sum module too short");
    let name = find_entry_point_name(&words).expect("sum must have entry point");
    assert_eq!(name, "main");
}

// ---- test_max_spirv_valid ----

#[test]
fn test_max_spirv_valid() {
    let bytes = generate_max_spirv(1024);
    assert_spirv_magic_bytes(&bytes, "max_1024");
    assert_eq!(bytes.len() % 4, 0, "max must be 4-byte aligned");
    let words = bytes_to_words(&bytes);
    assert!(words.len() >= 5, "max module too short");
    let name = find_entry_point_name(&words).expect("max must have entry point");
    assert_eq!(name, "main");
}

// ---- test_mean_spirv_valid ----

#[test]
fn test_mean_spirv_valid() {
    let bytes = generate_mean_spirv(1024);
    assert_spirv_magic_bytes(&bytes, "mean_1024");
    assert_eq!(bytes.len() % 4, 0, "mean must be 4-byte aligned");
    let words = bytes_to_words(&bytes);
    assert!(words.len() >= 5, "mean module too short");
    let name = find_entry_point_name(&words).expect("mean must have entry point");
    assert_eq!(name, "main");
}

// ---- test_softmax_spirv_valid ----

#[test]
fn test_softmax_spirv_valid() {
    let bytes = generate_softmax_spirv(32, 128);
    assert_spirv_magic_bytes(&bytes, "softmax_32x128");
    assert_eq!(bytes.len() % 4, 0, "softmax must be 4-byte aligned");
    let words = bytes_to_words(&bytes);
    assert!(words.len() >= 5, "softmax module too short");
    let name = find_entry_point_name(&words).expect("softmax must have entry point");
    assert_eq!(name, "main");
}

// ---- test_sum_spirv_workgroup_size ----

#[test]
fn test_sum_spirv_workgroup_size() {
    let bytes = generate_sum_spirv(256);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("sum must have workgroup size");
    assert_eq!(
        wg,
        [REDUCTION_WORKGROUP_SIZE, 1, 1],
        "sum workgroup size must match REDUCTION_WORKGROUP_SIZE"
    );
}

// ---- test_reduction_spirv_different_sizes ----

#[test]
fn test_reduction_spirv_different_sizes() {
    for n in [1, 128, 1024, 65536] {
        // Sum.
        let sum_bytes = generate_sum_spirv(n);
        assert_spirv_magic_bytes(&sum_bytes, &format!("sum_n={n}"));
        assert_eq!(sum_bytes.len() % 4, 0, "sum n={n}: must be 4-byte aligned");

        // Max.
        let max_bytes = generate_max_spirv(n);
        assert_spirv_magic_bytes(&max_bytes, &format!("max_n={n}"));
        assert_eq!(max_bytes.len() % 4, 0, "max n={n}: must be 4-byte aligned");

        // Mean.
        let mean_bytes = generate_mean_spirv(n);
        assert_spirv_magic_bytes(&mean_bytes, &format!("mean_n={n}"));
        assert_eq!(
            mean_bytes.len() % 4,
            0,
            "mean n={n}: must be 4-byte aligned"
        );

        // Validate headers via word inspection.
        let sum_words = bytes_to_words(&sum_bytes);
        let max_words = bytes_to_words(&max_bytes);
        let mean_words = bytes_to_words(&mean_bytes);
        assert!(sum_words.len() >= 5, "sum n={n}: module too short");
        assert!(max_words.len() >= 5, "max n={n}: module too short");
        assert!(mean_words.len() >= 5, "mean n={n}: module too short");
    }
}
