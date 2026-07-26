// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for CPU SIMD gather and scatter-add operations.

use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a deterministic input array: `input[i] = (i + 1) as f32 * 0.1`.
fn make_input(len: usize) -> Vec<f32> {
    (0..len).map(|i| (i + 1) as f32 * 0.1).collect()
}

// ===========================================================================
// gather_1d tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Basic gather correctness
// ---------------------------------------------------------------------------

#[test]
fn test_gather_scalar_sequential() {
    let input = make_input(10);
    let indices = [0u32, 1, 2, 3, 4];
    let mut output = vec![0.0f32; 5];

    gather_1d_scalar(&input, &indices, &mut output).expect("sequential gather should succeed");

    for (i, &idx) in indices.iter().enumerate() {
        assert_eq!(output[i], input[idx as usize], "position {i}");
    }
}

#[test]
fn test_gather_scalar_reverse() {
    let input = make_input(10);
    let indices = [9u32, 8, 7, 6, 5];
    let mut output = vec![0.0f32; 5];

    gather_1d_scalar(&input, &indices, &mut output).expect("reverse gather should succeed");

    for (i, &idx) in indices.iter().enumerate() {
        assert_eq!(output[i], input[idx as usize], "position {i}");
    }
}

#[test]
fn test_gather_scalar_duplicate_indices() {
    let input = make_input(5);
    let indices = [2u32, 2, 2, 4, 4];
    let mut output = vec![0.0f32; 5];

    gather_1d_scalar(&input, &indices, &mut output).expect("duplicate indices should succeed");

    assert_eq!(output[0], input[2]);
    assert_eq!(output[1], input[2]);
    assert_eq!(output[2], input[2]);
    assert_eq!(output[3], input[4]);
    assert_eq!(output[4], input[4]);
}

#[test]
fn test_gather_scalar_single_element() {
    let input = vec![42.0f32];
    let indices = [0u32];
    let mut output = vec![0.0f32; 1];

    gather_1d_scalar(&input, &indices, &mut output).expect("single element should succeed");
    assert_eq!(output[0], 42.0);
}

// ---------------------------------------------------------------------------
// gather_1d dispatch (SIMD auto-select)
// ---------------------------------------------------------------------------

#[test]
fn test_gather_dispatch_basic() {
    let input = make_input(20);
    let indices = [5u32, 10, 15, 0, 19];
    let mut output = vec![0.0f32; 5];

    gather_1d(&input, &indices, &mut output).expect("dispatch gather should succeed");

    for (i, &idx) in indices.iter().enumerate() {
        assert_eq!(output[i], input[idx as usize], "dispatch position {i}");
    }
}

#[test]
fn test_gather_dispatch_matches_scalar() {
    let input = make_input(100);
    let indices: Vec<u32> = (0..50).map(|i| (i * 2) as u32).collect();

    let mut scalar_out = vec![0.0f32; 50];
    gather_1d_scalar(&input, &indices, &mut scalar_out).expect("scalar should succeed");

    let mut dispatch_out = vec![0.0f32; 50];
    gather_1d(&input, &indices, &mut dispatch_out).expect("dispatch should succeed");

    assert_eq!(scalar_out, dispatch_out, "dispatch != scalar");
}

#[test]
fn test_gather_dispatch_matches_reference() {
    let input = make_input(100);
    let indices: Vec<u32> = (0..30).map(|i| (i * 3) as u32).collect();

    let reference = gather_1d_reference(&input, &indices).expect("reference should succeed");

    let mut dispatch_out = vec![0.0f32; 30];
    gather_1d(&input, &indices, &mut dispatch_out).expect("dispatch should succeed");

    assert_eq!(reference, dispatch_out, "dispatch != reference");
}

// ---------------------------------------------------------------------------
// gather_1d SIMD tail handling
// ---------------------------------------------------------------------------

#[test]
fn test_gather_varied_lengths() {
    let input = make_input(200);

    // Test various lengths to exercise SIMD chunks and scalar tails.
    for len in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100] {
        let indices: Vec<u32> = (0..len).map(|i| (i % 200) as u32).collect();

        let mut scalar_out = vec![0.0f32; len];
        gather_1d_scalar(&input, &indices, &mut scalar_out).expect("scalar should succeed");

        let mut dispatch_out = vec![0.0f32; len];
        gather_1d(&input, &indices, &mut dispatch_out).expect("dispatch should succeed");

        assert_eq!(scalar_out, dispatch_out, "mismatch at len={len}");
    }
}

// ---------------------------------------------------------------------------
// gather_1d known values
// ---------------------------------------------------------------------------

#[test]
fn test_gather_known_values() {
    let input = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let indices = [4u32, 0, 2, 1, 3];
    let mut output = vec![0.0f32; 5];

    gather_1d(&input, &indices, &mut output).expect("known values should succeed");

    assert_eq!(output, vec![50.0, 10.0, 30.0, 20.0, 40.0]);
}

// ---------------------------------------------------------------------------
// gather_1d error cases
// ---------------------------------------------------------------------------

#[test]
fn test_gather_oob_index() {
    let input = make_input(5);
    let indices = [0u32, 5]; // 5 is OOB for len=5
    let mut output = vec![0.0f32; 2];

    let result = gather_1d(&input, &indices, &mut output);
    match result {
        Err(GatherError::IndexOutOfBounds {
            index: 5,
            input_len: 5,
            position: 1,
        }) => {} // expected
        other => panic!("expected IndexOutOfBounds, got {other:?}"),
    }
}

#[test]
fn test_gather_oob_large_index() {
    let input = make_input(10);
    let indices = [u32::MAX];
    let mut output = vec![0.0f32; 1];

    let result = gather_1d(&input, &indices, &mut output);
    assert!(
        matches!(result, Err(GatherError::IndexOutOfBounds { .. })),
        "u32::MAX should be OOB"
    );
}

#[test]
fn test_gather_output_length_mismatch() {
    let input = make_input(10);
    let indices = [0u32, 1, 2];
    let mut output = vec![0.0f32; 2]; // should be 3

    let result = gather_1d(&input, &indices, &mut output);
    assert!(
        matches!(result, Err(GatherError::OutputLengthMismatch { .. })),
        "should reject mismatched output length"
    );
}

#[test]
fn test_gather_empty() {
    let input = make_input(10);
    let indices: &[u32] = &[];
    let mut output: Vec<f32> = vec![];

    gather_1d(&input, indices, &mut output).expect("empty gather should succeed");
    assert!(output.is_empty());
}

#[test]
fn test_gather_no_partial_writes_on_error() {
    let input = make_input(5);
    let indices = [0u32, 100]; // 100 is OOB
    let mut output = vec![f32::NAN; 2];

    let result = gather_1d_scalar(&input, &indices, &mut output);
    assert!(result.is_err());
    // Output should be untouched (fail-fast validation).
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_nan(), "output[{i}] was modified despite error");
    }
}

// ---------------------------------------------------------------------------
// gather_1d large batch
// ---------------------------------------------------------------------------

#[test]
fn test_gather_large_batch() {
    let input = make_input(10_000);
    let indices: Vec<u32> = (0..5000).map(|i| ((i * 7) % 10_000) as u32).collect();

    let reference = gather_1d_reference(&input, &indices).expect("reference should succeed");

    let mut dispatch_out = vec![0.0f32; 5000];
    gather_1d(&input, &indices, &mut dispatch_out).expect("dispatch should succeed");

    assert_eq!(reference, dispatch_out, "large batch mismatch");
}

// ===========================================================================
// scatter_add_1d tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Basic scatter-add correctness
// ---------------------------------------------------------------------------

#[test]
fn test_scatter_add_scalar_basic() {
    let input = vec![1.0, 2.0, 3.0];
    let indices = [0u32, 1, 2];
    let dim_size = 4;
    let mut output = vec![0.0f32; dim_size];

    scatter_add_1d_scalar(&input, &indices, dim_size, &mut output)
        .expect("basic scatter should succeed");

    assert_eq!(output, vec![1.0, 2.0, 3.0, 0.0]);
}

#[test]
fn test_scatter_add_scalar_accumulation() {
    // Multiple values scatter to the same bin.
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let indices = [0u32, 0, 1, 1];
    let dim_size = 3;
    let mut output = vec![0.0f32; dim_size];

    scatter_add_1d_scalar(&input, &indices, dim_size, &mut output)
        .expect("accumulation scatter should succeed");

    assert_eq!(output[0], 3.0); // 1.0 + 2.0
    assert_eq!(output[1], 7.0); // 3.0 + 4.0
    assert_eq!(output[2], 0.0); // untouched
}

#[test]
fn test_scatter_add_scalar_single_element() {
    let input = vec![42.0];
    let indices = [0u32];
    let dim_size = 1;
    let mut output = vec![0.0f32; dim_size];

    scatter_add_1d_scalar(&input, &indices, dim_size, &mut output)
        .expect("single element scatter should succeed");

    assert_eq!(output[0], 42.0);
}

#[test]
fn test_scatter_add_scalar_all_to_one_bin() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let indices = [0u32, 0, 0, 0, 0];
    let dim_size = 3;
    let mut output = vec![0.0f32; dim_size];

    scatter_add_1d_scalar(&input, &indices, dim_size, &mut output)
        .expect("all-to-one scatter should succeed");

    assert_eq!(output[0], 15.0); // 1+2+3+4+5
    assert_eq!(output[1], 0.0);
    assert_eq!(output[2], 0.0);
}

// ---------------------------------------------------------------------------
// scatter_add_1d dispatch
// ---------------------------------------------------------------------------

#[test]
fn test_scatter_add_dispatch_basic() {
    let input = vec![10.0, 20.0, 30.0];
    let indices = [2u32, 0, 1];
    let dim_size = 4;
    let mut output = vec![0.0f32; dim_size];

    scatter_add_1d(&input, &indices, dim_size, &mut output)
        .expect("dispatch scatter should succeed");

    assert_eq!(output, vec![20.0, 30.0, 10.0, 0.0]);
}

#[test]
fn test_scatter_add_dispatch_matches_scalar() {
    let input: Vec<f32> = (0..50).map(|i| i as f32 * 0.5).collect();
    let indices: Vec<u32> = (0..50).map(|i| (i % 20) as u32).collect();
    let dim_size = 20;

    let mut scalar_out = vec![0.0f32; dim_size];
    scatter_add_1d_scalar(&input, &indices, dim_size, &mut scalar_out)
        .expect("scalar should succeed");

    let mut dispatch_out = vec![0.0f32; dim_size];
    scatter_add_1d(&input, &indices, dim_size, &mut dispatch_out).expect("dispatch should succeed");

    assert_eq!(scalar_out, dispatch_out, "dispatch != scalar");
}

#[test]
fn test_scatter_add_dispatch_matches_reference() {
    let input: Vec<f32> = (0..30).map(|i| i as f32).collect();
    let indices: Vec<u32> = (0..30).map(|i| (i % 10) as u32).collect();
    let dim_size = 10;

    let reference =
        scatter_add_1d_reference(&input, &indices, dim_size).expect("reference should succeed");

    let mut dispatch_out = vec![0.0f32; dim_size];
    scatter_add_1d(&input, &indices, dim_size, &mut dispatch_out).expect("dispatch should succeed");

    assert_eq!(reference, dispatch_out, "dispatch != reference");
}

// ---------------------------------------------------------------------------
// scatter_add_1d with pre-filled output
// ---------------------------------------------------------------------------

#[test]
fn test_scatter_add_into_nonzero_output() {
    let input = vec![1.0, 2.0];
    let indices = [0u32, 1];
    let dim_size = 3;
    let mut output = vec![10.0, 20.0, 30.0];

    scatter_add_1d(&input, &indices, dim_size, &mut output)
        .expect("nonzero output scatter should succeed");

    assert_eq!(output, vec![11.0, 22.0, 30.0]);
}

// ---------------------------------------------------------------------------
// scatter_add_1d error cases
// ---------------------------------------------------------------------------

#[test]
fn test_scatter_add_oob_index() {
    let input = vec![1.0];
    let indices = [5u32]; // OOB for dim_size=3
    let dim_size = 3;
    let mut output = vec![0.0f32; dim_size];

    let result = scatter_add_1d(&input, &indices, dim_size, &mut output);
    match result {
        Err(GatherError::IndexOutOfBounds {
            index: 5,
            input_len: 3,
            position: 0,
        }) => {} // expected
        other => panic!("expected IndexOutOfBounds, got {other:?}"),
    }
}

#[test]
fn test_scatter_add_input_indices_length_mismatch() {
    let input = vec![1.0, 2.0];
    let indices = [0u32]; // length mismatch
    let dim_size = 3;
    let mut output = vec![0.0f32; dim_size];

    let result = scatter_add_1d(&input, &indices, dim_size, &mut output);
    assert!(
        matches!(result, Err(GatherError::OutputLengthMismatch { .. })),
        "should reject input/indices length mismatch"
    );
}

#[test]
fn test_scatter_add_output_dim_size_mismatch() {
    let input = vec![1.0];
    let indices = [0u32];
    let dim_size = 3;
    let mut output = vec![0.0f32; 5]; // should be 3

    let result = scatter_add_1d(&input, &indices, dim_size, &mut output);
    assert!(
        matches!(result, Err(GatherError::OutputLengthMismatch { .. })),
        "should reject output/dim_size mismatch"
    );
}

#[test]
fn test_scatter_add_empty() {
    let input: &[f32] = &[];
    let indices: &[u32] = &[];
    let dim_size = 5;
    let mut output = vec![0.0f32; dim_size];

    scatter_add_1d(input, indices, dim_size, &mut output).expect("empty scatter should succeed");

    assert_eq!(output, vec![0.0; 5]);
}

// ---------------------------------------------------------------------------
// scatter_add_1d large batch
// ---------------------------------------------------------------------------

#[test]
fn test_scatter_add_large_batch() {
    let input: Vec<f32> = (0..5000).map(|i| i as f32 * 0.01).collect();
    let indices: Vec<u32> = (0..5000).map(|i| (i % 100) as u32).collect();
    let dim_size = 100;

    let reference =
        scatter_add_1d_reference(&input, &indices, dim_size).expect("reference should succeed");

    let mut dispatch_out = vec![0.0f32; dim_size];
    scatter_add_1d(&input, &indices, dim_size, &mut dispatch_out).expect("dispatch should succeed");

    for (i, (&r, &d)) in reference.iter().zip(dispatch_out.iter()).enumerate() {
        assert!(
            (r - d).abs() < 1e-4,
            "large batch mismatch at bin {i}: reference={r}, dispatch={d}"
        );
    }
}

// ---------------------------------------------------------------------------
// Error display formatting
// ---------------------------------------------------------------------------

#[test]
fn test_error_display() {
    let e = GatherError::IndexOutOfBounds {
        index: 42,
        input_len: 30,
        position: 5,
    };
    let msg = format!("{e}");
    assert!(msg.contains("42"), "should contain index");
    assert!(msg.contains("30"), "should contain input_len");
    assert!(msg.contains("5"), "should contain position");

    let e2 = GatherError::OutputLengthMismatch {
        expected: 10,
        actual: 5,
    };
    let msg2 = format!("{e2}");
    assert!(msg2.contains("10"), "should contain expected");
    assert!(msg2.contains("5"), "should contain actual");
}

// ---------------------------------------------------------------------------
// GATHER_BLOCK_SIZE constant
// ---------------------------------------------------------------------------

#[test]
fn test_gather_block_size_is_power_of_two() {
    assert!(GATHER_BLOCK_SIZE.is_power_of_two());
    assert!(GATHER_BLOCK_SIZE >= 64);
}
