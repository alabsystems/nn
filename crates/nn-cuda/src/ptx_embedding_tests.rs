// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for CUDA C++ embedding lookup kernel generation.

use super::*;

// ---------------------------------------------------------------------------
// Config construction and validation
// ---------------------------------------------------------------------------

#[test]
fn test_config_default() {
    let c = PtxEmbeddingConfig::new(50257, 768);
    assert_eq!(c.vocab_size, 50257);
    assert_eq!(c.embedding_dim, 768);
    assert_eq!(c.block_size, 256);
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_custom_block_size() {
    let c = PtxEmbeddingConfig::new(50257, 768).with_block_size(128);
    assert_eq!(c.block_size, 128);
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_zero_vocab_rejected() {
    let c = PtxEmbeddingConfig::new(0, 768);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_zero_dim_rejected() {
    let c = PtxEmbeddingConfig::new(50257, 0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_zero_block_size_rejected() {
    let c = PtxEmbeddingConfig::new(50257, 768).with_block_size(0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_small_values() {
    let c = PtxEmbeddingConfig::new(2, 4);
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_large_vocab() {
    let c = PtxEmbeddingConfig::new(250_000, 1024);
    assert!(c.validate().is_ok());
}

// ---------------------------------------------------------------------------
// CUDA C++ output contains key patterns
// ---------------------------------------------------------------------------

#[test]
fn test_output_contains_global_keyword() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("__global__"),
        "kernel source must contain __global__ keyword"
    );
}

#[test]
fn test_output_contains_threadidx() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("threadIdx"),
        "kernel source must contain threadIdx"
    );
}

#[test]
fn test_output_contains_blockidx() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("blockIdx"),
        "kernel source must contain blockIdx"
    );
}

#[test]
fn test_output_contains_griddim() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("gridDim"),
        "kernel source must use gridDim for grid-stride pattern"
    );
}

#[test]
fn test_output_contains_blockdim() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("blockDim"),
        "kernel source must use blockDim for grid-stride pattern"
    );
}

#[test]
fn test_output_contains_restrict_pointers() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    assert_eq!(
        src.matches("__restrict__").count(),
        3,
        "must have 3 __restrict__ pointers (token_ids, embedding_table, output)"
    );
}

#[test]
fn test_output_contains_grid_stride_loop() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    // Grid-stride pattern: idx += gridDim.x * blockDim.x
    assert!(
        src.contains("idx += gridDim.x * blockDim.x"),
        "must contain grid-stride advance expression"
    );
}

#[test]
fn test_output_contains_kernel_name() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("embedding_lookup"),
        "kernel function must be named embedding_lookup"
    );
}

#[test]
fn test_output_contains_bounds_check() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("tok < vocab_size"),
        "must bounds-check token ID against vocab_size"
    );
}

#[test]
fn test_output_contains_zero_fill() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("0.0f"),
        "must zero-fill out-of-bounds token output"
    );
}

#[test]
fn test_output_contains_embedding_table_param() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("embedding_table"),
        "must reference embedding_table parameter"
    );
}

#[test]
fn test_output_contains_token_ids_param() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("token_ids"),
        "must reference token_ids parameter"
    );
}

#[test]
fn test_output_contains_vocab_size_param() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("vocab_size"),
        "must reference vocab_size parameter"
    );
}

#[test]
fn test_output_contains_embedding_dim_param() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("embedding_dim"),
        "must reference embedding_dim parameter"
    );
}

// ---------------------------------------------------------------------------
// Header comment documents configuration
// ---------------------------------------------------------------------------

#[test]
fn test_header_contains_vocab_size() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("vocab_size=50257"),
        "header must document vocab_size"
    );
}

#[test]
fn test_header_contains_embedding_dim() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("embedding_dim=768"),
        "header must document embedding_dim"
    );
}

#[test]
fn test_header_contains_block_size() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("block_size=256"),
        "header must document block_size"
    );
}

// ---------------------------------------------------------------------------
// Different vocab/dim sizes produce different output
// ---------------------------------------------------------------------------

#[test]
fn test_different_vocab_sizes() {
    let src_small = generate_embedding_ptx(&PtxEmbeddingConfig::new(1000, 768)).unwrap();
    let src_large = generate_embedding_ptx(&PtxEmbeddingConfig::new(50257, 768)).unwrap();
    assert_ne!(
        src_small, src_large,
        "different vocab_size should produce different output"
    );
}

#[test]
fn test_different_embedding_dims() {
    let src_512 = generate_embedding_ptx(&PtxEmbeddingConfig::new(50257, 512)).unwrap();
    let src_768 = generate_embedding_ptx(&PtxEmbeddingConfig::new(50257, 768)).unwrap();
    assert_ne!(
        src_512, src_768,
        "different embedding_dim should produce different output"
    );
}

#[test]
fn test_different_block_sizes() {
    let src_128 =
        generate_embedding_ptx(&PtxEmbeddingConfig::new(50257, 768).with_block_size(128)).unwrap();
    let src_256 = generate_embedding_ptx(&PtxEmbeddingConfig::new(50257, 768)).unwrap();
    assert_ne!(
        src_128, src_256,
        "different block_size should produce different output"
    );
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_small_vocab_small_dim() {
    let config = PtxEmbeddingConfig::new(2, 1);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(src.contains("__global__"));
    assert!(src.contains("vocab_size=2"));
    assert!(src.contains("embedding_dim=1"));
}

#[test]
fn test_large_vocab_large_dim() {
    let config = PtxEmbeddingConfig::new(250_000, 4096);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(src.contains("vocab_size=250000"));
    assert!(src.contains("embedding_dim=4096"));
}

#[test]
fn test_custom_block_size_in_header() {
    let config = PtxEmbeddingConfig::new(50257, 768).with_block_size(512);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("block_size=512"),
        "header must show custom block_size"
    );
}

// ---------------------------------------------------------------------------
// Launch config math
// ---------------------------------------------------------------------------

#[test]
fn test_launch_config_basic() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let (grid, block) = ptx_embedding_launch_config(32, &config);
    // total = 32 * 768 = 24576, grid = ceil(24576 / 256) = 96
    assert_eq!(grid, 96);
    assert_eq!(block, 256);
}

#[test]
fn test_launch_config_single_token() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let (grid, block) = ptx_embedding_launch_config(1, &config);
    // total = 1 * 768 = 768, grid = ceil(768 / 256) = 3
    assert_eq!(grid, 3);
    assert_eq!(block, 256);
}

#[test]
fn test_launch_config_small_dim() {
    let config = PtxEmbeddingConfig::new(100, 64);
    let (grid, block) = ptx_embedding_launch_config(10, &config);
    // total = 10 * 64 = 640, grid = ceil(640 / 256) = 3
    assert_eq!(grid, 3);
    assert_eq!(block, 256);
}

#[test]
fn test_launch_config_exact_multiple() {
    let config = PtxEmbeddingConfig::new(50257, 256);
    let (grid, block) = ptx_embedding_launch_config(4, &config);
    // total = 4 * 256 = 1024, grid = ceil(1024 / 256) = 4
    assert_eq!(grid, 4);
    assert_eq!(block, 256);
}

#[test]
fn test_launch_config_not_exact_multiple() {
    let config = PtxEmbeddingConfig::new(50257, 100);
    let (grid, block) = ptx_embedding_launch_config(3, &config);
    // total = 3 * 100 = 300, grid = ceil(300 / 256) = 2
    assert_eq!(grid, 2);
    assert_eq!(block, 256);
}

#[test]
fn test_launch_config_custom_block_size() {
    let config = PtxEmbeddingConfig::new(50257, 768).with_block_size(128);
    let (grid, block) = ptx_embedding_launch_config(32, &config);
    // total = 32 * 768 = 24576, grid = ceil(24576 / 128) = 192
    assert_eq!(grid, 192);
    assert_eq!(block, 128);
}

#[test]
fn test_launch_config_large_batch() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let (grid, block) = ptx_embedding_launch_config(1024, &config);
    // total = 1024 * 768 = 786432, grid = ceil(786432 / 256) = 3072
    assert_eq!(grid, 3072);
    assert_eq!(block, 256);
}

#[test]
fn test_launch_config_dim_1() {
    let config = PtxEmbeddingConfig::new(100, 1);
    let (grid, block) = ptx_embedding_launch_config(5, &config);
    // total = 5 * 1 = 5, grid = ceil(5 / 256) = 1
    assert_eq!(grid, 1);
    assert_eq!(block, 256);
}

// ---------------------------------------------------------------------------
// Kernel structure validation
// ---------------------------------------------------------------------------

#[test]
fn test_kernel_is_complete_function() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    // Opening brace after parameter list
    let open_count = src.matches('{').count();
    let close_count = src.matches('}').count();
    assert_eq!(
        open_count, close_count,
        "braces must be balanced: {open_count} open, {close_count} close"
    );
}

#[test]
fn test_kernel_has_for_loop() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("for ("),
        "grid-stride loop must use for-loop syntax"
    );
}

#[test]
fn test_output_index_arithmetic() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    // The kernel must compute output index from token_idx and dim_idx
    assert!(
        src.contains("token_idx * embedding_dim + dim_idx"),
        "must compute flat output index as token_idx * embedding_dim + dim_idx"
    );
}

#[test]
fn test_table_index_arithmetic() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    // The kernel must index into embedding table
    assert!(
        src.contains("tok * embedding_dim + dim_idx"),
        "must compute table index as tok * embedding_dim + dim_idx"
    );
}

// ---------------------------------------------------------------------------
// Validation rejects bad configs via generate_embedding_ptx
// ---------------------------------------------------------------------------

#[test]
fn test_generate_rejects_zero_vocab() {
    let config = PtxEmbeddingConfig::new(0, 768);
    assert!(generate_embedding_ptx(&config).is_err());
}

#[test]
fn test_generate_rejects_zero_dim() {
    let config = PtxEmbeddingConfig::new(50257, 0);
    assert!(generate_embedding_ptx(&config).is_err());
}

#[test]
fn test_generate_rejects_zero_block_size() {
    let config = PtxEmbeddingConfig::new(50257, 768).with_block_size(0);
    assert!(generate_embedding_ptx(&config).is_err());
}

// ---------------------------------------------------------------------------
// EMBEDDING_BLOCK_SIZE constant
// ---------------------------------------------------------------------------

#[test]
fn test_embedding_block_size_constant() {
    assert_eq!(EMBEDDING_BLOCK_SIZE, 256);
}

// ---------------------------------------------------------------------------
// CPU reference: embedding_reference
// ---------------------------------------------------------------------------

#[test]
fn test_reference_single_token() {
    // table: 3 rows, dim=2: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    let table = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let out = embedding_reference(&[1], &table, 2);
    assert_eq!(out, vec![3.0, 4.0]);
}

#[test]
fn test_reference_batch_lookup() {
    let table = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // 2 rows, dim=3
    let out = embedding_reference(&[0, 1, 0], &table, 3);
    assert_eq!(out.len(), 9);
    assert_eq!(&out[0..3], &[0.1, 0.2, 0.3]);
    assert_eq!(&out[3..6], &[0.4, 0.5, 0.6]);
    assert_eq!(&out[6..9], &[0.1, 0.2, 0.3]);
}

#[test]
fn test_reference_boundary_indices() {
    // vocab_size=3, dim=2
    let table = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    // Index 2 is the last valid row
    let out = embedding_reference(&[2], &table, 2);
    assert_eq!(out, vec![50.0, 60.0]);
    // Index 0 is the first valid row
    let out0 = embedding_reference(&[0], &table, 2);
    assert_eq!(out0, vec![10.0, 20.0]);
}

#[test]
fn test_reference_oov_produces_zeros() {
    let table = vec![1.0, 2.0, 3.0, 4.0]; // 2 rows, dim=2
                                          // Index 5 is out of bounds (vocab_size=2)
    let out = embedding_reference(&[5], &table, 2);
    assert_eq!(out, vec![0.0, 0.0]);
}

#[test]
fn test_reference_mixed_valid_and_oov() {
    let table = vec![1.0, 2.0, 3.0, 4.0]; // 2 rows, dim=2
    let out = embedding_reference(&[0, 99, 1], &table, 2);
    assert_eq!(&out[0..2], &[1.0, 2.0]); // valid
    assert_eq!(&out[2..4], &[0.0, 0.0]); // OOV
    assert_eq!(&out[4..6], &[3.0, 4.0]); // valid
}

#[test]
fn test_reference_empty_indices() {
    let table = vec![1.0, 2.0];
    let out = embedding_reference(&[], &table, 2);
    assert!(out.is_empty());
}

#[test]
fn test_reference_dim_1() {
    let table = vec![10.0, 20.0, 30.0]; // 3 rows, dim=1
    let out = embedding_reference(&[2, 0, 1], &table, 1);
    assert_eq!(out, vec![30.0, 10.0, 20.0]);
}

#[test]
fn test_reference_correctness_large_table() {
    // Construct a table where row i = [i*10.0, i*10.0+1, i*10.0+2, i*10.0+3]
    let dim = 4;
    let vocab = 100;
    let table: Vec<f32> = (0..vocab)
        .flat_map(|i| (0..dim).map(move |d| (i * 10 + d) as f32))
        .collect();
    let indices: Vec<u32> = vec![0, 50, 99];
    let out = embedding_reference(&indices, &table, dim);
    assert_eq!(&out[0..4], &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(&out[4..8], &[500.0, 501.0, 502.0, 503.0]);
    assert_eq!(&out[8..12], &[990.0, 991.0, 992.0, 993.0]);
}
