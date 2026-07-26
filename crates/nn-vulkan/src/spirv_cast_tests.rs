// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SPIR-V dtype cast kernels (F32/F16/BF16 conversions).
//!
//! Covers:
//! - SPIR-V structural validity (header, magic, version, generator)
//! - Float16 / Int16 capability presence
//! - Entry point naming and workgroup size
//! - BF16 bitwise roundtrip precision
//! - Special values (zero, infinity, NaN)

use super::*;
use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};
use crate::spirv_emit::SPIRV_MAGIC;

// SPIR-V constants for structural assertions.
const TEST_SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const TEST_GENERATOR_MAGIC: u32 = 0x4E4E_0000;
const TEST_OP_CAPABILITY: u16 = 17;
const TEST_CAPABILITY_FLOAT16: u32 = 9;
const TEST_CAPABILITY_INT16: u32 = 22;
const TEST_CAPABILITY_SHADER: u32 = 1;

// ---- Helpers ----

fn assert_valid_header(words: &[u32], label: &str) {
    assert!(words.len() >= 5, "{label}: module too short");
    assert_eq!(
        words[0], SPIRV_MAGIC,
        "{label}: wrong magic (expected 0x07230203)"
    );
    assert_eq!(
        words[1], TEST_SPIRV_VERSION_1_0,
        "{label}: wrong version (expected 1.0)"
    );
    assert_eq!(
        words[2], TEST_GENERATOR_MAGIC,
        "{label}: wrong generator magic"
    );
}

fn has_capability(words: &[u32], capability: u32) -> bool {
    let mut i = 5; // skip header
    while i < words.len() {
        let word = words[i];
        let wc = (word >> 16) as usize;
        let opc = (word & 0xFFFF) as u16;
        if opc == TEST_OP_CAPABILITY && wc >= 2 && i + 1 < words.len() {
            if words[i + 1] == capability {
                return true;
            }
        }
        if wc == 0 {
            break;
        }
        i += wc;
    }
    false
}

/// BF16 conversion helpers for test validation.
fn f32_to_bf16_bits(val: f32) -> u16 {
    (val.to_bits() >> 16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

// ---- F32 -> F16 tests ----

#[test]
fn test_f32_to_f16_spirv_header() {
    let spirv = generate_f32_to_f16_spirv(1024);
    assert_valid_header(&spirv, "f32_to_f16");
}

#[test]
fn test_f32_to_f16_spirv_entry_point() {
    let spirv = generate_f32_to_f16_spirv(1024);
    let name = find_entry_point_name(&spirv);
    assert_eq!(
        name.as_deref(),
        Some("main"),
        "f32_to_f16 entry point should be 'main'"
    );
}

#[test]
fn test_f32_to_f16_spirv_workgroup_size() {
    let spirv = generate_f32_to_f16_spirv(1024);
    let wg = find_workgroup_size(&spirv);
    assert_eq!(
        wg,
        Some([CAST_WORKGROUP_SIZE, 1, 1]),
        "f32_to_f16 workgroup size should be [{CAST_WORKGROUP_SIZE}, 1, 1]"
    );
}

#[test]
fn test_f32_to_f16_spirv_has_float16_capability() {
    let spirv = generate_f32_to_f16_spirv(512);
    assert!(
        has_capability(&spirv, TEST_CAPABILITY_FLOAT16),
        "f32_to_f16 should declare Float16 capability"
    );
}

#[test]
fn test_f32_to_f16_spirv_has_shader_capability() {
    let spirv = generate_f32_to_f16_spirv(512);
    assert!(
        has_capability(&spirv, TEST_CAPABILITY_SHADER),
        "f32_to_f16 should declare Shader capability"
    );
}

// ---- F16 -> F32 tests ----

#[test]
fn test_f16_to_f32_spirv_header() {
    let spirv = generate_f16_to_f32_spirv(1024);
    assert_valid_header(&spirv, "f16_to_f32");
}

#[test]
fn test_f16_to_f32_spirv_entry_point() {
    let spirv = generate_f16_to_f32_spirv(1024);
    let name = find_entry_point_name(&spirv);
    assert_eq!(
        name.as_deref(),
        Some("main"),
        "f16_to_f32 entry point should be 'main'"
    );
}

#[test]
fn test_f16_to_f32_spirv_workgroup_size() {
    let spirv = generate_f16_to_f32_spirv(1024);
    let wg = find_workgroup_size(&spirv);
    assert_eq!(
        wg,
        Some([CAST_WORKGROUP_SIZE, 1, 1]),
        "f16_to_f32 workgroup size should be [{CAST_WORKGROUP_SIZE}, 1, 1]"
    );
}

#[test]
fn test_f16_to_f32_spirv_has_float16_capability() {
    let spirv = generate_f16_to_f32_spirv(512);
    assert!(
        has_capability(&spirv, TEST_CAPABILITY_FLOAT16),
        "f16_to_f32 should declare Float16 capability"
    );
}

// ---- F32 -> BF16 tests ----

#[test]
fn test_f32_to_bf16_spirv_header() {
    let spirv = generate_f32_to_bf16_spirv(1024);
    assert_valid_header(&spirv, "f32_to_bf16");
}

#[test]
fn test_f32_to_bf16_spirv_entry_point() {
    let spirv = generate_f32_to_bf16_spirv(1024);
    let name = find_entry_point_name(&spirv);
    assert_eq!(
        name.as_deref(),
        Some("main"),
        "f32_to_bf16 entry point should be 'main'"
    );
}

#[test]
fn test_f32_to_bf16_spirv_workgroup_size() {
    let spirv = generate_f32_to_bf16_spirv(1024);
    let wg = find_workgroup_size(&spirv);
    assert_eq!(
        wg,
        Some([CAST_WORKGROUP_SIZE, 1, 1]),
        "f32_to_bf16 workgroup size should be [{CAST_WORKGROUP_SIZE}, 1, 1]"
    );
}

#[test]
fn test_f32_to_bf16_spirv_has_int16_capability() {
    let spirv = generate_f32_to_bf16_spirv(512);
    assert!(
        has_capability(&spirv, TEST_CAPABILITY_INT16),
        "f32_to_bf16 should declare Int16 capability"
    );
}

// ---- BF16 -> F32 tests ----

#[test]
fn test_bf16_to_f32_spirv_header() {
    let spirv = generate_bf16_to_f32_spirv(1024);
    assert_valid_header(&spirv, "bf16_to_f32");
}

#[test]
fn test_bf16_to_f32_spirv_entry_point() {
    let spirv = generate_bf16_to_f32_spirv(1024);
    let name = find_entry_point_name(&spirv);
    assert_eq!(
        name.as_deref(),
        Some("main"),
        "bf16_to_f32 entry point should be 'main'"
    );
}

#[test]
fn test_bf16_to_f32_spirv_workgroup_size() {
    let spirv = generate_bf16_to_f32_spirv(1024);
    let wg = find_workgroup_size(&spirv);
    assert_eq!(
        wg,
        Some([CAST_WORKGROUP_SIZE, 1, 1]),
        "bf16_to_f32 workgroup size should be [{CAST_WORKGROUP_SIZE}, 1, 1]"
    );
}

#[test]
fn test_bf16_to_f32_spirv_has_int16_capability() {
    let spirv = generate_bf16_to_f32_spirv(512);
    assert!(
        has_capability(&spirv, TEST_CAPABILITY_INT16),
        "bf16_to_f32 should declare Int16 capability"
    );
}

// ---- BF16 precision tests (reference computations) ----

#[test]
fn test_bf16_roundtrip_precision() {
    // BF16 has 8-bit mantissa (7 explicit bits + 1 implicit). Values that fit
    // exactly in bf16 should roundtrip perfectly.
    let values = [0.0_f32, 1.0, -1.0, 2.0, 0.5, 128.0, -256.0];
    for &v in &values {
        let bits = f32_to_bf16_bits(v);
        let recovered = bf16_bits_to_f32(bits);
        assert_eq!(
            v, recovered,
            "bf16 roundtrip failed for {v}: bits=0x{bits:04x}, recovered={recovered}"
        );
    }
}

#[test]
fn test_bf16_precision_loss() {
    // 1.001 cannot be exactly represented in bf16 — precision loss expected.
    let v = 1.001_f32;
    let bits = f32_to_bf16_bits(v);
    let recovered = bf16_bits_to_f32(bits);
    // The error should be within bf16 precision (~1/128 = 0.0078).
    let err = (v - recovered).abs();
    assert!(
        err < 0.01,
        "bf16 precision loss too large for {v}: recovered={recovered}, err={err}"
    );
    // But it should NOT be exact.
    assert_ne!(v, recovered, "1.001 should lose precision in bf16");
}

#[test]
fn test_bf16_special_values() {
    // Zero roundtrips.
    assert_eq!(bf16_bits_to_f32(f32_to_bf16_bits(0.0)), 0.0);
    assert_eq!(bf16_bits_to_f32(f32_to_bf16_bits(-0.0)), -0.0);

    // Infinity roundtrips.
    assert_eq!(
        bf16_bits_to_f32(f32_to_bf16_bits(f32::INFINITY)),
        f32::INFINITY
    );
    assert_eq!(
        bf16_bits_to_f32(f32_to_bf16_bits(f32::NEG_INFINITY)),
        f32::NEG_INFINITY
    );

    // NaN: bits should produce a NaN (not necessarily the same NaN payload).
    let nan_bits = f32_to_bf16_bits(f32::NAN);
    let nan_recovered = bf16_bits_to_f32(nan_bits);
    assert!(nan_recovered.is_nan(), "NaN should roundtrip as NaN");
}

// ---- Different sizes tests ----

#[test]
fn test_cast_different_sizes() {
    for n in [1, 64, 256, 1024, 4096] {
        let spirv = generate_f32_to_f16_spirv(n);
        assert!(spirv.len() > 5, "f32_to_f16 n={n}: module too short");
        assert_eq!(spirv[0], SPIRV_MAGIC, "f32_to_f16 n={n}: wrong magic");

        let spirv = generate_bf16_to_f32_spirv(n);
        assert!(spirv.len() > 5, "bf16_to_f32 n={n}: module too short");
        assert_eq!(spirv[0], SPIRV_MAGIC, "bf16_to_f32 n={n}: wrong magic");
    }
}
