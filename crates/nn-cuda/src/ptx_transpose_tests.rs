// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PTX transpose kernel generation.

use super::*;

// ---------------------------------------------------------------------------
// PTX generation: 2D transpose
// ---------------------------------------------------------------------------

#[test]
fn test_generate_transpose_ptx_contains_entry_and_shared_memory() {
    let ptx = generate_transpose_ptx(64, 128);
    assert!(
        ptx.contains(".entry ptx_transpose_f32"),
        "missing kernel entry"
    );
    assert!(ptx.contains(".shared"), "missing shared memory declaration");
    assert!(ptx.contains("tile_smem"), "missing tile_smem");
    assert!(ptx.contains("bar.sync"), "missing barrier sync");
    assert!(ptx.contains(".target sm_70"), "missing SM target");
    assert!(
        ptx.contains(".address_size 64"),
        "missing 64-bit addressing"
    );
}

#[test]
fn test_generate_transpose_ptx_has_params() {
    let ptx = generate_transpose_ptx(32, 32);
    assert!(ptx.contains("param_input"), "missing input param");
    assert!(ptx.contains("param_output"), "missing output param");
    assert!(ptx.contains("param_rows"), "missing rows param");
    assert!(ptx.contains("param_cols"), "missing cols param");
}

#[test]
fn test_generate_transpose_ptx_shared_memory_padded() {
    let ptx = generate_transpose_ptx(16, 16);
    // TILE=16, smem should be TILE*(TILE+1) = 16*17 = 272 floats
    assert!(
        ptx.contains("tile_smem[272]"),
        "shared memory should be padded: TILE * (TILE+1)"
    );
}

// ---------------------------------------------------------------------------
// PTX generation: batched transpose
// ---------------------------------------------------------------------------

#[test]
fn test_generate_batch_transpose_ptx_contains_batch_param() {
    let ptx = generate_batch_transpose_ptx(8, 64, 128);
    assert!(
        ptx.contains(".entry ptx_batch_transpose_f32"),
        "missing kernel entry"
    );
    assert!(ptx.contains("param_batch"), "missing batch param");
    assert!(
        ptx.contains("%ctaid.z"),
        "missing z-dimension batch indexing"
    );
}

#[test]
fn test_generate_batch_transpose_ptx_has_shared_memory() {
    let ptx = generate_batch_transpose_ptx(4, 32, 64);
    assert!(ptx.contains(".shared"), "missing shared memory");
    assert!(ptx.contains("bar.sync"), "missing barrier sync");
}

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------

#[test]
fn test_ptx_transpose_launch_config_exact_tile() {
    let (grid, block) = ptx_transpose_launch_config(16, 16);
    assert_eq!(block, [16, 16, 1]);
    assert_eq!(grid, [1, 1, 1]);
}

#[test]
fn test_ptx_transpose_launch_config_multi_tile() {
    let (grid, block) = ptx_transpose_launch_config(64, 128);
    assert_eq!(block, [16, 16, 1]);
    assert_eq!(grid, [8, 4, 1]); // ceil(128/16)=8, ceil(64/16)=4
}

#[test]
fn test_ptx_transpose_launch_config_non_aligned() {
    let (grid, block) = ptx_transpose_launch_config(17, 33);
    assert_eq!(block, [16, 16, 1]);
    assert_eq!(grid, [3, 2, 1]); // ceil(33/16)=3, ceil(17/16)=2
}

#[test]
fn test_ptx_batch_transpose_launch_config() {
    let (grid, block) = ptx_batch_transpose_launch_config(4, 32, 64);
    assert_eq!(block, [16, 16, 1]);
    assert_eq!(grid, [4, 2, 4]); // ceil(64/16)=4, ceil(32/16)=2, batch=4
}

// ---------------------------------------------------------------------------
// Reference implementations
// ---------------------------------------------------------------------------

#[test]
fn test_transpose_reference_square() {
    // 2x2 matrix
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = transpose_reference(&data, 2, 2);
    assert_eq!(result, vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_transpose_reference_rectangular() {
    // 2x3 matrix -> 3x2
    // Input:  [[1, 2, 3],
    //          [4, 5, 6]]
    // Output: [[1, 4],
    //          [2, 5],
    //          [3, 6]]
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = transpose_reference(&data, 2, 3);
    assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_reference_single_row() {
    // 1x4 -> 4x1
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = transpose_reference(&data, 1, 4);
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_transpose_reference_single_col() {
    // 4x1 -> 1x4
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = transpose_reference(&data, 4, 1);
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_transpose_reference_involutory() {
    // Transpose of transpose is identity
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t1 = transpose_reference(&data, 2, 3);
    let t2 = transpose_reference(&t1, 3, 2);
    assert_eq!(t2, data);
}

#[test]
fn test_batch_transpose_reference() {
    // Batch of 2, each 2x3
    let data = vec![
        // matrix 0: [[1,2,3],[4,5,6]]
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // matrix 1: [[7,8,9],[10,11,12]]
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let result = batch_transpose_reference(&data, 2, 2, 3);
    assert_eq!(
        result,
        vec![
            // matrix 0 transposed: [[1,4],[2,5],[3,6]]
            1.0, 4.0, 2.0, 5.0, 3.0, 6.0, // matrix 1 transposed: [[7,10],[8,11],[9,12]]
            7.0, 10.0, 8.0, 11.0, 9.0, 12.0,
        ]
    );
}

#[test]
fn test_batch_transpose_reference_involutory() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t1 = batch_transpose_reference(&data, 2, 3, 4);
    let t2 = batch_transpose_reference(&t1, 2, 4, 3);
    assert_eq!(t2, data);
}
