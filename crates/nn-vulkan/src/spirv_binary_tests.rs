// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! External tests for `spirv_binary` module.
//!
//! Tests the public API from outside the module, exercising SPIR-V binary
//! generation for add, mul, relu, scalar_mul, and transpose kernels, as well
//! as the `find_entry_point_name` and `find_workgroup_size` utility functions.

use crate::spirv_binary::{
    emit_add_spirv, emit_mul_spirv, emit_relu_spirv, emit_scalar_mul_spirv, emit_transpose_spirv,
    find_entry_point_name, find_workgroup_size, BINARY_WORKGROUP_SIZE,
};
use crate::spirv_emit::SPIRV_MAGIC;

/// Assert that a SPIR-V word stream has a valid header.
fn assert_valid_spirv(spirv: &[u32], label: &str) {
    assert!(spirv.len() >= 5, "{label}: module too short for header");
    assert_eq!(spirv[0], SPIRV_MAGIC, "{label}: wrong SPIR-V magic number");
    // Version 1.0 = 0x00010000.
    assert_eq!(
        spirv[1], 0x0001_0000,
        "{label}: expected SPIR-V 1.0 version"
    );
    // Bound (word[3]) must be positive — it is the upper bound on ID values.
    assert!(spirv[3] > 0, "{label}: ID bound must be > 0");
    // Schema (word[4]) must be 0 per spec.
    assert_eq!(spirv[4], 0, "{label}: schema must be 0");
}

// ---- BINARY_WORKGROUP_SIZE constant ----

#[test]
fn test_binary_workgroup_size_is_power_of_two() {
    assert!(
        BINARY_WORKGROUP_SIZE.is_power_of_two(),
        "BINARY_WORKGROUP_SIZE ({BINARY_WORKGROUP_SIZE}) must be a power of 2"
    );
}

#[test]
fn test_binary_workgroup_size_within_limits() {
    assert!(
        BINARY_WORKGROUP_SIZE <= 1024,
        "BINARY_WORKGROUP_SIZE ({BINARY_WORKGROUP_SIZE}) exceeds Vulkan guaranteed minimum (1024)"
    );
    assert!(
        BINARY_WORKGROUP_SIZE > 0,
        "BINARY_WORKGROUP_SIZE must be > 0"
    );
}

// ---- emit_add_spirv ----

#[test]
fn test_add_spirv_valid_module() {
    let spirv = emit_add_spirv().expect("emit_add_spirv must succeed");
    assert_valid_spirv(&spirv, "add");
}

#[test]
fn test_add_spirv_entry_point_is_main() {
    let spirv = emit_add_spirv().unwrap();
    let name = find_entry_point_name(&spirv).expect("add: must have entry point");
    assert_eq!(name, "main", "add: entry point must be 'main'");
}

#[test]
fn test_add_spirv_workgroup_size() {
    let spirv = emit_add_spirv().unwrap();
    let wg = find_workgroup_size(&spirv).expect("add: must have workgroup size");
    assert_eq!(
        wg,
        [BINARY_WORKGROUP_SIZE, 1, 1],
        "add: workgroup size must be [BINARY_WORKGROUP_SIZE, 1, 1]"
    );
}

// ---- emit_mul_spirv ----

#[test]
fn test_mul_spirv_valid_module() {
    let spirv = emit_mul_spirv().expect("emit_mul_spirv must succeed");
    assert_valid_spirv(&spirv, "mul");
}

#[test]
fn test_mul_spirv_entry_point_is_main() {
    let spirv = emit_mul_spirv().unwrap();
    let name = find_entry_point_name(&spirv).expect("mul: must have entry point");
    assert_eq!(name, "main", "mul: entry point must be 'main'");
}

#[test]
fn test_mul_spirv_workgroup_size() {
    let spirv = emit_mul_spirv().unwrap();
    let wg = find_workgroup_size(&spirv).expect("mul: must have workgroup size");
    assert_eq!(wg, [BINARY_WORKGROUP_SIZE, 1, 1]);
}

// ---- emit_relu_spirv ----

#[test]
fn test_relu_spirv_valid_module() {
    let spirv = emit_relu_spirv().expect("emit_relu_spirv must succeed");
    assert_valid_spirv(&spirv, "relu");
}

#[test]
fn test_relu_spirv_entry_point_is_main() {
    let spirv = emit_relu_spirv().unwrap();
    let name = find_entry_point_name(&spirv).expect("relu: must have entry point");
    assert_eq!(name, "main", "relu: entry point must be 'main'");
}

#[test]
fn test_relu_spirv_workgroup_size() {
    let spirv = emit_relu_spirv().unwrap();
    let wg = find_workgroup_size(&spirv).expect("relu: must have workgroup size");
    assert_eq!(wg, [BINARY_WORKGROUP_SIZE, 1, 1]);
}

// ---- emit_scalar_mul_spirv ----

#[test]
fn test_scalar_mul_spirv_valid_module() {
    let spirv = emit_scalar_mul_spirv().expect("emit_scalar_mul_spirv must succeed");
    assert_valid_spirv(&spirv, "scalar_mul");
}

#[test]
fn test_scalar_mul_spirv_entry_point_is_main() {
    let spirv = emit_scalar_mul_spirv().unwrap();
    let name = find_entry_point_name(&spirv).expect("scalar_mul: must have entry point");
    assert_eq!(name, "main", "scalar_mul: entry point must be 'main'");
}

#[test]
fn test_scalar_mul_spirv_workgroup_size() {
    let spirv = emit_scalar_mul_spirv().unwrap();
    let wg = find_workgroup_size(&spirv).expect("scalar_mul: must have workgroup size");
    assert_eq!(wg, [BINARY_WORKGROUP_SIZE, 1, 1]);
}

// ---- emit_transpose_spirv ----

#[test]
fn test_transpose_spirv_valid_module() {
    let spirv = emit_transpose_spirv().expect("emit_transpose_spirv must succeed");
    assert_valid_spirv(&spirv, "transpose");
}

#[test]
fn test_transpose_spirv_entry_point_is_main() {
    let spirv = emit_transpose_spirv().unwrap();
    let name = find_entry_point_name(&spirv).expect("transpose: must have entry point");
    assert_eq!(name, "main", "transpose: entry point must be 'main'");
}

#[test]
fn test_transpose_spirv_workgroup_size() {
    let spirv = emit_transpose_spirv().unwrap();
    let wg = find_workgroup_size(&spirv).expect("transpose: must have workgroup size");
    assert_eq!(wg, [BINARY_WORKGROUP_SIZE, 1, 1]);
}

// ---- find_entry_point_name edge cases ----

#[test]
fn test_find_entry_point_name_empty_module() {
    assert_eq!(find_entry_point_name(&[]), None);
}

#[test]
fn test_find_entry_point_name_wrong_magic() {
    // 5-word header with wrong magic.
    assert_eq!(find_entry_point_name(&[0xDEADBEEF, 0, 0, 0, 0]), None);
}

#[test]
fn test_find_entry_point_name_too_short() {
    // Fewer than 5 words.
    assert_eq!(find_entry_point_name(&[SPIRV_MAGIC, 0, 0, 0]), None);
}

// ---- find_workgroup_size edge cases ----

#[test]
fn test_find_workgroup_size_empty_module() {
    assert_eq!(find_workgroup_size(&[]), None);
}

#[test]
fn test_find_workgroup_size_wrong_magic() {
    assert_eq!(find_workgroup_size(&[0xDEADBEEF, 0, 0, 0, 0]), None);
}

#[test]
fn test_find_workgroup_size_too_short() {
    assert_eq!(find_workgroup_size(&[SPIRV_MAGIC, 0, 0, 0]), None);
}

// ---- Cross-cutting: all emitters produce distinct modules ----

#[test]
fn test_all_emitters_produce_nonempty_modules() {
    let emitters: Vec<(&str, Vec<u32>)> = vec![
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
        ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
        ("transpose", emit_transpose_spirv().unwrap()),
    ];
    for (name, spirv) in &emitters {
        assert!(
            spirv.len() > 50,
            "{name}: module too small ({} words)",
            spirv.len()
        );
        assert!(
            spirv.len() < 1000,
            "{name}: module too large ({} words)",
            spirv.len()
        );
    }
}

#[test]
fn test_all_emitters_consistent_workgroup_size() {
    // All binary emitters must use the same BINARY_WORKGROUP_SIZE.
    let emitters: Vec<(&str, Vec<u32>)> = vec![
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
        ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
        ("transpose", emit_transpose_spirv().unwrap()),
    ];
    for (name, spirv) in &emitters {
        let wg = find_workgroup_size(spirv)
            .unwrap_or_else(|| panic!("{name}: must have workgroup size"));
        assert_eq!(
            wg[0], BINARY_WORKGROUP_SIZE,
            "{name}: X workgroup size mismatch"
        );
        assert_eq!(wg[1], 1, "{name}: Y workgroup size must be 1");
        assert_eq!(wg[2], 1, "{name}: Z workgroup size must be 1");
    }
}
