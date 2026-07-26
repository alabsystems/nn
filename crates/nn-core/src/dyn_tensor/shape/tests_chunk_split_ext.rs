// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for `chunk` and `split` shape operations:
//! value verification, edge cases, 2D/3D inputs.

use crate::dyn_tensor::test_helpers::{cpu, t1d, t2d};
use crate::DynTensor;

// -- chunk value verification -------------------------------------------------

#[test]
fn test_chunk_values_1d() {
    let t = t1d(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let chunks = t.chunk(3, 0).unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].to_flat_vec::<f32>().unwrap(), vec![10.0, 20.0]);
    assert_eq!(chunks[1].to_flat_vec::<f32>().unwrap(), vec![30.0, 40.0]);
    assert_eq!(chunks[2].to_flat_vec::<f32>().unwrap(), vec![50.0, 60.0]);
}

#[test]
fn test_chunk_uneven_values() {
    // 5 elements into 3 chunks: [2, 2, 1]
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let chunks = t.chunk(3, 0).unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(chunks[1].to_flat_vec::<f32>().unwrap(), vec![3.0, 4.0]);
    assert_eq!(chunks[2].to_flat_vec::<f32>().unwrap(), vec![5.0]);
}

#[test]
fn test_chunk_single_chunk() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let chunks = t.chunk(1, 0).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].dims(), &[3]);
    assert_eq!(chunks[0].to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_chunk_more_chunks_than_elements() {
    // 3 elements into 5 chunks: should produce 3 chunks of size 1
    let t = t1d(&[1.0, 2.0, 3.0]);
    let chunks = t.chunk(5, 0).unwrap();
    assert_eq!(chunks.len(), 3);
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.dims(), &[1], "chunk {i} should have size 1");
    }
}

#[test]
fn test_chunk_2d_dim1() {
    // [2, 4] tensor, chunk along dim=1 into 2 pieces
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2, 4);
    let chunks = t.chunk(2, 1).unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].dims(), &[2, 2]);
    assert_eq!(chunks[1].dims(), &[2, 2]);
    assert_eq!(
        chunks[0].to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 5.0, 6.0]
    );
    assert_eq!(
        chunks[1].to_flat_vec::<f32>().unwrap(),
        vec![3.0, 4.0, 7.0, 8.0]
    );
}

// -- split value verification for 2D dim1 -------------------------------------

#[test]
fn test_split_2d_dim1_values() {
    let t = DynTensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        &cpu(),
    )
    .unwrap();
    let parts = t.split([1, 2, 1], 1).unwrap();
    assert_eq!(parts.len(), 3);
    // Part 0: column 0
    assert_eq!(parts[0].to_flat_vec::<f32>().unwrap(), vec![1.0, 5.0, 9.0]);
    // Part 1: columns 1-2
    assert_eq!(
        parts[1].to_flat_vec::<f32>().unwrap(),
        vec![2.0, 3.0, 6.0, 7.0, 10.0, 11.0]
    );
    // Part 2: column 3
    assert_eq!(parts[2].to_flat_vec::<f32>().unwrap(), vec![4.0, 8.0, 12.0]);
}

// -- chunk and cat roundtrip --------------------------------------------------

#[test]
fn test_chunk_cat_roundtrip() {
    let original = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let chunks = original.chunk(3, 0).unwrap();
    let refs: Vec<&DynTensor> = chunks.iter().collect();
    let reconstructed = DynTensor::cat(&refs, 0).unwrap();
    assert_eq!(
        reconstructed.to_flat_vec::<f32>().unwrap(),
        original.to_flat_vec::<f32>().unwrap()
    );
}
