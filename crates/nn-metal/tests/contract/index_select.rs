// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Contract tests for IndexSelect and Gather compiled-pipeline dispatch.
//!
//! Verifies:
//! - Correct row selection via uint indices (AC1, AC2 of #2278)
//! - OOB index clamping to last valid row (AC3 of #2278)
//! - Gather per-element index lookup
//!
//! Tests the full pipeline: IR → dispatch plan → MSL codegen → Metal execution.
//! Part of #2278.

use super::test_utils::{metal_setup, rand_f32_vec};

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::ScalarType;
use nn_metal::execute_tensor_dispatch;
use std::collections::HashMap;

/// Helper: assert `actual ≈ expected` within tolerance.
fn assert_close(label: &str, actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 1e-6,
        "{label}: expected {expected}, got {actual}",
    );
}

// ===========================================================================
// IndexSelect tests
// ===========================================================================

/// IndexSelect: select 3 rows from a [10, 4] tensor along dim 0.
/// GPU output must match exact row lookups.
/// Part of #2278.
#[test]
fn test_index_select_basic_correctness() {
    let (num_rows, cols) = (10, 4);
    let num_indices = 3;

    let mut b = TensorBlockBuilder::new("isel_basic");
    let data = b.add_input("data", &[num_rows, cols]);
    let idx = b.add_input("idx", &[num_indices]);
    let out = b.add_index_select(data, idx, 0, &[num_indices, cols]);
    let def = b.build(out).expect("valid graph");

    let cache = metal_setup();

    let data_vec: Vec<f32> = (0..num_rows * cols).map(|i| i as f32 * 0.1).collect();
    let idx_vec: Vec<f32> = vec![2.0, 7.0, 0.0];

    let mut inputs = HashMap::new();
    inputs.insert("data", data_vec.clone());
    inputs.insert("idx", idx_vec);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("index_select GPU dispatch");
    assert_eq!(gpu_out.len(), num_indices * cols, "output length");

    let expected_rows: [usize; 3] = [2, 7, 0];
    for (i, &row) in expected_rows.iter().enumerate() {
        for c in 0..cols {
            assert_close(
                &format!("index_select[{i}][{c}]"),
                gpu_out[i * cols + c],
                data_vec[row * cols + c],
            );
        }
    }
}

/// IndexSelect: OOB index is clamped to last valid row.
/// An index >= dim_size should read from the last row, not crash or read garbage.
/// Part of #2278 (AC3).
#[test]
fn test_index_select_oob_clamp() {
    let (num_rows, cols) = (5, 3);
    let num_indices = 4;

    let mut b = TensorBlockBuilder::new("isel_oob");
    let data = b.add_input("data", &[num_rows, cols]);
    let idx = b.add_input("idx", &[num_indices]);
    let out = b.add_index_select(data, idx, 0, &[num_indices, cols]);
    let def = b.build(out).expect("valid graph");

    let cache = metal_setup();

    let data_vec: Vec<f32> = (0..num_rows * cols).map(|i| (i + 1) as f32).collect();
    // Indices: row 0 (valid), row 4 (last valid), row 999 (OOB), row 5 (OOB).
    let idx_vec: Vec<f32> = vec![0.0, 4.0, 999.0, 5.0];

    let mut inputs = HashMap::new();
    inputs.insert("data", data_vec.clone());
    inputs.insert("idx", idx_vec);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("index_select OOB GPU dispatch");
    assert_eq!(gpu_out.len(), num_indices * cols);

    let last_row = num_rows - 1;

    // expected_row_idx for each output row: [0, 4, 4(clamped), 4(clamped)]
    let expected_rows = [0usize, last_row, last_row, last_row];
    for (out_row, &src_row) in expected_rows.iter().enumerate() {
        for c in 0..cols {
            assert_close(
                &format!("oob[{out_row}][{c}]"),
                gpu_out[out_row * cols + c],
                data_vec[src_row * cols + c],
            );
        }
    }
}

/// IndexSelect: select along dim 1 (inner dimension).
/// Verifies the 3-way decomposition handles non-zero outer dimensions.
/// Part of #2278.
#[test]
fn test_index_select_dim1() {
    let (rows, cols) = (3, 8);
    let num_indices = 2;

    let mut b = TensorBlockBuilder::new("isel_dim1");
    let data = b.add_input("data", &[rows, cols]);
    let idx = b.add_input("idx", &[num_indices]);
    let out = b.add_index_select(data, idx, 1, &[rows, num_indices]);
    let def = b.build(out).expect("valid graph");

    let cache = metal_setup();

    let data_vec: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    let idx_vec: Vec<f32> = vec![5.0, 1.0];

    let mut inputs = HashMap::new();
    inputs.insert("data", data_vec.clone());
    inputs.insert("idx", idx_vec);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("index_select dim1 GPU dispatch");
    assert_eq!(gpu_out.len(), rows * num_indices);

    let selected_cols = [5usize, 1];
    for r in 0..rows {
        for (i, &col) in selected_cols.iter().enumerate() {
            assert_close(
                &format!("dim1[{r}][{i}]"),
                gpu_out[r * num_indices + i],
                data_vec[r * cols + col],
            );
        }
    }
}

// ===========================================================================
// Gather tests
// ===========================================================================

/// Gather: gather along dim 1 from [2, 5] with [2, 3] indices → [2, 3].
/// Part of #2278.
#[test]
fn test_gather_basic_correctness() {
    let (rows, cols) = (2, 5);
    let gather_cols = 3;

    let mut b = TensorBlockBuilder::new("gather_basic");
    let data = b.add_input("data", &[rows, cols]);
    let idx = b.add_input("idx", &[rows, gather_cols]);
    let out = b.add_gather(data, idx, 1, &[rows, gather_cols]);
    let def = b.build(out).expect("valid graph");

    let cache = metal_setup();

    let data_vec: Vec<f32> = (0..rows * cols).map(|i| (i + 1) as f32 * 0.5).collect();
    // Row 0: [4, 0, 2], Row 1: [1, 3, 4]
    let idx_vec: Vec<f32> = vec![4.0, 0.0, 2.0, 1.0, 3.0, 4.0];

    let idx_u: Vec<usize> = idx_vec.iter().map(|v| *v as usize).collect();
    let mut inputs = HashMap::new();
    inputs.insert("data", data_vec.clone());
    inputs.insert("idx", idx_vec);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("gather GPU dispatch");
    assert_eq!(gpu_out.len(), rows * gather_cols);

    for r in 0..rows {
        for c in 0..gather_cols {
            let src_col = idx_u[r * gather_cols + c];
            assert_close(
                &format!("gather[{r}][{c}]"),
                gpu_out[r * gather_cols + c],
                data_vec[r * cols + src_col],
            );
        }
    }
}

/// Gather: OOB indices are clamped.
/// Part of #2278 (AC3).
#[test]
fn test_gather_oob_clamp() {
    let (rows, cols) = (2, 4);
    let gather_cols = 2;

    let mut b = TensorBlockBuilder::new("gather_oob");
    let data = b.add_input("data", &[rows, cols]);
    let idx = b.add_input("idx", &[rows, gather_cols]);
    let out = b.add_gather(data, idx, 1, &[rows, gather_cols]);
    let def = b.build(out).expect("valid graph");

    let cache = metal_setup();

    let data_vec: Vec<f32> = (0..rows * cols).map(|i| (i + 1) as f32).collect();
    // Row 0: [1 (valid), 100 (OOB → clamp to col 3)]
    // Row 1: [0 (valid), 4 (OOB → clamp to col 3)]
    let idx_vec: Vec<f32> = vec![1.0, 100.0, 0.0, 4.0];

    let mut inputs = HashMap::new();
    inputs.insert("data", data_vec.clone());
    inputs.insert("idx", idx_vec);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("gather OOB GPU dispatch");
    assert_eq!(gpu_out.len(), rows * gather_cols);

    let last_col = cols - 1;

    // (out_flat_idx, expected_row, expected_col) tuples
    let checks: [(usize, usize, usize); 4] = [
        (0, 0, 1),        // Row 0, idx=1 → data[0][1]
        (1, 0, last_col), // Row 0, idx=100 → clamped to data[0][3]
        (2, 1, 0),        // Row 1, idx=0 → data[1][0]
        (3, 1, last_col), // Row 1, idx=4 → clamped to data[1][3]
    ];
    for &(flat, r, c) in &checks {
        assert_close(
            &format!("gather_oob[{flat}]"),
            gpu_out[flat],
            data_vec[r * cols + c],
        );
    }
}

/// IndexSelect: negative float indices are clamped to 0 in f32→u32 conversion.
/// MSL `uint(negative_float)` is implementation-defined; the conversion kernel
/// explicitly guards with `(v < 0.0f) ? 0u : uint(v)`.
/// Part of #2278.
#[test]
fn test_index_select_negative_indices_clamp_to_zero() {
    let (num_rows, cols) = (4, 3);
    let num_indices = 3;

    let mut b = TensorBlockBuilder::new("isel_neg");
    let data = b.add_input("data", &[num_rows, cols]);
    let idx = b.add_input("idx", &[num_indices]);
    let out = b.add_index_select(data, idx, 0, &[num_indices, cols]);
    let def = b.build(out).expect("valid graph");

    let cache = metal_setup();

    let data_vec: Vec<f32> = (0..num_rows * cols).map(|i| (i + 1) as f32).collect();
    // Negative indices: all should clamp to row 0.
    let idx_vec: Vec<f32> = vec![-1.0, -100.0, -0.5];

    let mut inputs = HashMap::new();
    inputs.insert("data", data_vec.clone());
    inputs.insert("idx", idx_vec);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("index_select negative-index GPU dispatch");
    assert_eq!(gpu_out.len(), num_indices * cols);

    // All negative indices should resolve to row 0.
    for i in 0..num_indices {
        for c in 0..cols {
            assert_close(
                &format!("neg_idx[{i}][{c}]"),
                gpu_out[i * cols + c],
                data_vec[c], // row 0
            );
        }
    }
}

/// IndexSelect with larger dimensions to exercise the f32→u32 conversion pipeline.
/// Part of #2278.
#[test]
fn test_index_select_larger_dims() {
    let (num_rows, cols) = (64, 16);
    let num_indices = 32;

    let mut b = TensorBlockBuilder::new("isel_large");
    let data = b.add_input("data", &[num_rows, cols]);
    let idx = b.add_input("idx", &[num_indices]);
    let out = b.add_index_select(data, idx, 0, &[num_indices, cols]);
    let def = b.build(out).expect("valid graph");

    let cache = metal_setup();

    let data_vec = rand_f32_vec(0x2278_0001, num_rows * cols, -1.0, 1.0);
    let idx_vec: Vec<f32> = rand_f32_vec(0x2278_0002, num_indices, 0.0, 63.0)
        .iter()
        .map(|v| v.floor())
        .collect();

    let mut inputs = HashMap::new();
    inputs.insert("data", data_vec.clone());
    inputs.insert("idx", idx_vec.clone());

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("index_select larger GPU dispatch");
    assert_eq!(gpu_out.len(), num_indices * cols);

    for (i, &idx_f) in idx_vec.iter().enumerate() {
        let row = idx_f as usize;
        for c in 0..cols {
            assert_close(
                &format!("large_isel[{i}][{c}]"),
                gpu_out[i * cols + c],
                data_vec[row * cols + c],
            );
        }
    }
}
