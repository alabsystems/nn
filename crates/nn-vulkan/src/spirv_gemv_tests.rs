// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::generate_gemv_spirv`], [`super::generate_dot_spirv`],
//! and [`super::generate_outer_spirv`].

use super::*;
use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};
use crate::spirv_emit::SPIRV_MAGIC;

// ---- Helpers ----

fn assert_valid_header(spirv: &[u32], label: &str) {
    assert!(spirv.len() >= 5, "{label}: module too short");
    assert_eq!(spirv[0], SPIRV_MAGIC, "{label}: wrong magic number");
    assert_eq!(spirv[1], SPIRV_VERSION_1_0, "{label}: wrong SPIR-V version");
    assert_eq!(spirv[2], GENERATOR_MAGIC, "{label}: wrong generator magic");
    assert!(spirv[3] > 0, "{label}: bound must be > 0");
    assert_eq!(spirv[4], 0, "{label}: schema must be 0");
}

fn has_opcode(spirv: &[u32], target_opcode: u16) -> bool {
    let mut pos = 5;
    while pos < spirv.len() {
        let word = spirv[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if word_count == 0 || pos + word_count > spirv.len() {
            break;
        }
        if opcode == target_opcode {
            return true;
        }
        pos += word_count;
    }
    false
}

// ---- GEMV SPIR-V header validation ----

#[test]
fn test_gemv_spirv_valid_header() {
    let spirv = generate_gemv_spirv(128, 256);
    assert_valid_header(&spirv, "gemv(128,256)");
}

#[test]
fn test_gemv_spirv_entry_point_main() {
    let spirv = generate_gemv_spirv(64, 128);
    let name =
        find_entry_point_name(&spirv).unwrap_or_else(|| panic!("gemv: no entry point found"));
    assert_eq!(name, "main", "gemv: entry point must be 'main'");
}

#[test]
fn test_gemv_spirv_workgroup_size() {
    let spirv = generate_gemv_spirv(64, 128);
    let wg = find_workgroup_size(&spirv).unwrap_or_else(|| panic!("gemv: no workgroup size found"));
    assert_eq!(
        wg,
        [GEMV_WORKGROUP_SIZE, 1, 1],
        "gemv: workgroup size must be [{GEMV_WORKGROUP_SIZE}, 1, 1]",
    );
}

#[test]
fn test_gemv_spirv_has_loop_structure() {
    let spirv = generate_gemv_spirv(64, 128);
    assert!(
        has_opcode(&spirv, OP_LOOP_MERGE),
        "gemv: must have OpLoopMerge for reduction loop"
    );
    assert!(
        has_opcode(&spirv, OP_PHI),
        "gemv: must have OpPhi for accumulator"
    );
}

#[test]
fn test_gemv_spirv_deterministic() {
    let a = generate_gemv_spirv(128, 256);
    let b = generate_gemv_spirv(128, 256);
    assert_eq!(a, b, "same params must produce identical SPIR-V");
}

#[test]
fn test_gemv_spirv_various_sizes() {
    let configs: &[(u32, u32)] = &[(1, 1), (4, 8), (64, 64), (256, 512), (1024, 768)];
    for &(m, n) in configs {
        let spirv = generate_gemv_spirv(m, n);
        let label = format!("gemv({m},{n})");
        assert_valid_header(&spirv, &label);
        let name =
            find_entry_point_name(&spirv).unwrap_or_else(|| panic!("{label}: no entry point"));
        assert_eq!(name, "main", "{label}: wrong entry point");
    }
}

// ---- GEMV reference tests ----

#[test]
fn test_gemv_reference_identity() {
    // Identity matrix: y = I @ x = x
    let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let x = vec![3.0, 5.0, 7.0];
    let y = gemv_reference(&a, &x, 3, 3);
    assert_eq!(y.len(), 3);
    assert!((y[0] - 3.0).abs() < 1e-6);
    assert!((y[1] - 5.0).abs() < 1e-6);
    assert!((y[2] - 7.0).abs() < 1e-6);
}

#[test]
fn test_gemv_reference_non_square() {
    // 2x3 matrix times 3-vector
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = vec![1.0, 1.0, 1.0];
    let y = gemv_reference(&a, &x, 2, 3);
    assert_eq!(y.len(), 2);
    // row 0: 1+2+3 = 6
    assert!((y[0] - 6.0).abs() < 1e-6);
    // row 1: 4+5+6 = 15
    assert!((y[1] - 15.0).abs() < 1e-6);
}

#[test]
fn test_gemv_reference_single_row() {
    // 1x4 matrix (row vector dot product)
    let a = vec![2.0, 3.0, 4.0, 5.0];
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = gemv_reference(&a, &x, 1, 4);
    assert_eq!(y.len(), 1);
    // 2*1 + 3*2 + 4*3 + 5*4 = 2+6+12+20 = 40
    assert!((y[0] - 40.0).abs() < 1e-6);
}

// ---- Dot product SPIR-V tests ----

#[test]
fn test_dot_spirv_valid_header() {
    let spirv = generate_dot_spirv(1024);
    assert_valid_header(&spirv, "dot(1024)");
}

#[test]
fn test_dot_spirv_entry_point_main() {
    let spirv = generate_dot_spirv(256);
    let name = find_entry_point_name(&spirv).unwrap_or_else(|| panic!("dot: no entry point found"));
    assert_eq!(name, "main", "dot: entry point must be 'main'");
}

#[test]
fn test_dot_spirv_workgroup_size() {
    let spirv = generate_dot_spirv(256);
    let wg = find_workgroup_size(&spirv).unwrap_or_else(|| panic!("dot: no workgroup size found"));
    assert_eq!(
        wg,
        [GEMV_WORKGROUP_SIZE, 1, 1],
        "dot: workgroup size must be [{GEMV_WORKGROUP_SIZE}, 1, 1]",
    );
}

#[test]
fn test_dot_spirv_has_fmul() {
    let spirv = generate_dot_spirv(128);
    assert!(
        has_opcode(&spirv, OP_FMUL),
        "dot: must have FMul for element-wise products"
    );
}

#[test]
fn test_dot_spirv_deterministic() {
    let a = generate_dot_spirv(512);
    let b = generate_dot_spirv(512);
    assert_eq!(a, b, "same params must produce identical SPIR-V");
}

// ---- Dot product reference tests ----

#[test]
fn test_dot_reference_basic() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let result = dot_reference(&a, &b);
    // 1*4 + 2*5 + 3*6 = 4+10+18 = 32
    assert!((result - 32.0).abs() < 1e-6);
}

#[test]
fn test_dot_reference_orthogonal() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let result = dot_reference(&a, &b);
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_dot_reference_single_element() {
    let a = vec![7.0];
    let b = vec![3.0];
    let result = dot_reference(&a, &b);
    assert!((result - 21.0).abs() < 1e-6);
}

// ---- Outer product SPIR-V tests ----

#[test]
fn test_outer_spirv_valid_header() {
    let spirv = generate_outer_spirv(64, 32);
    assert_valid_header(&spirv, "outer(64,32)");
}

#[test]
fn test_outer_spirv_entry_point_main() {
    let spirv = generate_outer_spirv(64, 32);
    let name =
        find_entry_point_name(&spirv).unwrap_or_else(|| panic!("outer: no entry point found"));
    assert_eq!(name, "main", "outer: entry point must be 'main'");
}

#[test]
fn test_outer_spirv_workgroup_size() {
    let spirv = generate_outer_spirv(64, 32);
    let wg =
        find_workgroup_size(&spirv).unwrap_or_else(|| panic!("outer: no workgroup size found"));
    assert_eq!(
        wg,
        [GEMV_WORKGROUP_SIZE, 1, 1],
        "outer: workgroup size must be [{GEMV_WORKGROUP_SIZE}, 1, 1]",
    );
}

#[test]
fn test_outer_spirv_has_fmul() {
    let spirv = generate_outer_spirv(64, 32);
    assert!(
        has_opcode(&spirv, OP_FMUL),
        "outer: must have FMul for element-wise products"
    );
}

#[test]
fn test_outer_spirv_deterministic() {
    let a = generate_outer_spirv(128, 64);
    let b = generate_outer_spirv(128, 64);
    assert_eq!(a, b, "same params must produce identical SPIR-V");
}

// ---- Outer product reference tests ----

#[test]
fn test_outer_reference_basic() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0];
    let c = outer_reference(&a, &b);
    assert_eq!(c.len(), 6);
    // Row 0: [1*4, 1*5] = [4, 5]
    assert!((c[0] - 4.0).abs() < 1e-6);
    assert!((c[1] - 5.0).abs() < 1e-6);
    // Row 1: [2*4, 2*5] = [8, 10]
    assert!((c[2] - 8.0).abs() < 1e-6);
    assert!((c[3] - 10.0).abs() < 1e-6);
    // Row 2: [3*4, 3*5] = [12, 15]
    assert!((c[4] - 12.0).abs() < 1e-6);
    assert!((c[5] - 15.0).abs() < 1e-6);
}

#[test]
fn test_outer_reference_single_elements() {
    let a = vec![3.0];
    let b = vec![7.0];
    let c = outer_reference(&a, &b);
    assert_eq!(c.len(), 1);
    assert!((c[0] - 21.0).abs() < 1e-6);
}

#[test]
fn test_outer_reference_zeros() {
    let a = vec![0.0, 1.0];
    let b = vec![5.0, 0.0, 3.0];
    let c = outer_reference(&a, &b);
    assert_eq!(c.len(), 6);
    // Row 0: [0*5, 0*0, 0*3] = [0, 0, 0]
    assert!((c[0] - 0.0).abs() < 1e-6);
    assert!((c[1] - 0.0).abs() < 1e-6);
    assert!((c[2] - 0.0).abs() < 1e-6);
    // Row 1: [1*5, 1*0, 1*3] = [5, 0, 3]
    assert!((c[3] - 5.0).abs() < 1e-6);
    assert!((c[4] - 0.0).abs() < 1e-6);
    assert!((c[5] - 3.0).abs() < 1e-6);
}

// ---- Cross-operation consistency ----

#[test]
fn test_gemv_dot_consistency() {
    // A single-row GEMV should be equivalent to dot product.
    let a = vec![2.0, 3.0, 4.0];
    let x = vec![1.0, 2.0, 3.0];

    let gemv_result = gemv_reference(&a, &x, 1, 3);
    let dot_result = dot_reference(&a, &x);

    assert!(
        (gemv_result[0] - dot_result).abs() < 1e-6,
        "single-row GEMV ({}) should match dot product ({})",
        gemv_result[0],
        dot_result,
    );
}

// ---- All three SPIR-V generators produce distinct modules ----

#[test]
fn test_all_spirv_modules_distinct() {
    let gemv = generate_gemv_spirv(64, 64);
    let dot = generate_dot_spirv(64);
    let outer = generate_outer_spirv(64, 64);

    assert_ne!(gemv, dot, "GEMV and dot should produce different SPIR-V");
    assert_ne!(
        gemv, outer,
        "GEMV and outer should produce different SPIR-V"
    );
    assert_ne!(dot, outer, "dot and outer should produce different SPIR-V");
}

// ---- Workgroup size constant value ----

#[test]
fn test_gemv_workgroup_size_constant() {
    assert_eq!(GEMV_WORKGROUP_SIZE, 256);
}
