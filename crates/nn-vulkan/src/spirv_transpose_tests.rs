// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! External tests for `spirv_transpose` module.
//!
//! Tests the public API from outside the module, exercising SPIR-V generation
//! for 2D and batched transpose kernels, plus CPU reference correctness.

use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};
use crate::spirv_emit::SPIRV_MAGIC;
use crate::spirv_transpose::{
    generate_batch_transpose_spirv, generate_transpose_spirv, transpose_reference,
    TRANSPOSE_WORKGROUP_SIZE,
};

// ---- test_transpose_spirv_valid ----

#[test]
fn test_transpose_spirv_valid() {
    let words = generate_transpose_spirv(8, 8);
    assert!(words.len() >= 5, "transpose module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "transpose: wrong SPIR-V magic");
    let name = find_entry_point_name(&words).expect("transpose must have entry point");
    assert_eq!(name, "main");
}

// ---- test_batch_transpose_spirv_valid ----

#[test]
fn test_batch_transpose_spirv_valid() {
    let words = generate_batch_transpose_spirv(4, 8, 8);
    assert!(words.len() >= 5, "batch transpose module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "batch transpose: wrong SPIR-V magic");
    let name = find_entry_point_name(&words).expect("batch transpose must have entry point");
    assert_eq!(name, "main");
    let wg = find_workgroup_size(&words).expect("batch transpose must have workgroup size");
    assert_eq!(wg, [TRANSPOSE_WORKGROUP_SIZE, TRANSPOSE_WORKGROUP_SIZE, 1]);
}

// ---- test_transpose_reference_square ----

#[test]
fn test_transpose_reference_square() {
    // 4x4 matrix transpose.
    #[rustfmt::skip]
    let data = vec![
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let result = transpose_reference(&data, 4, 4);
    #[rustfmt::skip]
    let expected = vec![
        1.0, 5.0, 9.0,  13.0,
        2.0, 6.0, 10.0, 14.0,
        3.0, 7.0, 11.0, 15.0,
        4.0, 8.0, 12.0, 16.0,
    ];
    assert_eq!(result, expected, "4x4 transpose mismatch");
}

// ---- test_transpose_reference_rectangular ----

#[test]
fn test_transpose_reference_rectangular() {
    // 3x5 matrix transpose -> 5x3.
    #[rustfmt::skip]
    let data = vec![
        1.0,  2.0,  3.0,  4.0,  5.0,
        6.0,  7.0,  8.0,  9.0,  10.0,
        11.0, 12.0, 13.0, 14.0, 15.0,
    ];
    let result = transpose_reference(&data, 3, 5);
    // Expected: 5x3 stored row-major.
    #[rustfmt::skip]
    let expected = vec![
        1.0, 6.0, 11.0,
        2.0, 7.0, 12.0,
        3.0, 8.0, 13.0,
        4.0, 9.0, 14.0,
        5.0, 10.0, 15.0,
    ];
    assert_eq!(result, expected, "3x5 transpose mismatch");
}

// ---- test_transpose_reference_identity ----

#[test]
fn test_transpose_reference_identity() {
    // 1x1 transpose is identity.
    let data = vec![42.0];
    let result = transpose_reference(&data, 1, 1);
    assert_eq!(result, vec![42.0], "1x1 transpose must be identity");
}

// ---- test_batch_transpose_reference ----

#[test]
fn test_batch_transpose_reference() {
    // Batched transpose: two 2x3 matrices.
    // Batch 0: [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    // Batch 1: [[7,8,9],[10,11,12]] -> [[7,10],[8,11],[9,12]]
    let batch0 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let batch1 = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let t0 = transpose_reference(&batch0, 2, 3);
    let t1 = transpose_reference(&batch1, 2, 3);

    // Batch 0 transposed: 3x2 row-major.
    assert_eq!(t0, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    // Batch 1 transposed: 3x2 row-major.
    assert_eq!(t1, vec![7.0, 10.0, 8.0, 11.0, 9.0, 12.0]);

    // Also verify the batch kernel generates valid SPIR-V.
    let words = generate_batch_transpose_spirv(2, 2, 3);
    assert!(words.len() >= 5, "batch transpose 2x2x3 module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "batch transpose: wrong magic");
}
