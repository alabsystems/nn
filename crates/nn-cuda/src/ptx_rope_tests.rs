// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for CUDA C++ RoPE kernel generation and CPU reference.

use super::*;

// =========================================================================
// Config construction and validation
// =========================================================================

#[test]
fn test_config_default() {
    let c = PtxRopeConfig::new(512, 64);
    assert_eq!(c.seq_len, 512);
    assert_eq!(c.head_dim, 64);
    assert_eq!(c.block_size, 256);
    assert_eq!(c.base, 10000.0);
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_custom_block_size() {
    let c = PtxRopeConfig::new(512, 64).with_block_size(128);
    assert_eq!(c.block_size, 128);
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_custom_base() {
    let c = PtxRopeConfig::new(512, 64).with_base(500000.0);
    assert_eq!(c.base, 500000.0);
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_zero_seq_len_rejected() {
    let c = PtxRopeConfig::new(0, 64);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_zero_head_dim_rejected() {
    let c = PtxRopeConfig::new(512, 0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_odd_head_dim_rejected() {
    let c = PtxRopeConfig::new(512, 63);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_zero_block_size_rejected() {
    let c = PtxRopeConfig::new(512, 64).with_block_size(0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_nan_base_rejected() {
    let c = PtxRopeConfig::new(512, 64).with_base(f32::NAN);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_negative_base_rejected() {
    let c = PtxRopeConfig::new(512, 64).with_base(-1.0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_inf_base_rejected() {
    let c = PtxRopeConfig::new(512, 64).with_base(f32::INFINITY);
    assert!(c.validate().is_err());
}

// =========================================================================
// On-the-fly kernel: structural validation
// =========================================================================

#[test]
fn test_rope_contains_global_keyword() {
    let config = PtxRopeConfig::new(512, 64);
    let src = generate_rope_ptx(&config).unwrap();
    assert!(src.contains("__global__"));
}

#[test]
fn test_rope_contains_kernel_name() {
    let config = PtxRopeConfig::new(512, 64);
    let src = generate_rope_ptx(&config).unwrap();
    assert!(src.contains("rope_apply"));
}

#[test]
fn test_rope_contains_trig_intrinsics() {
    let config = PtxRopeConfig::new(512, 64);
    let src = generate_rope_ptx(&config).unwrap();
    assert!(src.contains("__cosf"), "must use fast cosine intrinsic");
    assert!(src.contains("__sinf"), "must use fast sine intrinsic");
}

#[test]
fn test_rope_contains_grid_stride_loop() {
    let config = PtxRopeConfig::new(512, 64);
    let src = generate_rope_ptx(&config).unwrap();
    assert!(src.contains("gridDim.x * blockDim.x"));
}

#[test]
fn test_rope_contains_restrict_pointers() {
    let config = PtxRopeConfig::new(512, 64);
    let src = generate_rope_ptx(&config).unwrap();
    assert_eq!(
        src.matches("__restrict__").count(),
        2,
        "must have 2 __restrict__ pointers (input, output)"
    );
}

#[test]
fn test_rope_header_documents_config() {
    let config = PtxRopeConfig::new(512, 64);
    let src = generate_rope_ptx(&config).unwrap();
    assert!(src.contains("seq_len=512"));
    assert!(src.contains("head_dim=64"));
    assert!(src.contains("block_size=256"));
}

#[test]
fn test_rope_contains_powf_base() {
    let config = PtxRopeConfig::new(512, 64);
    let src = generate_rope_ptx(&config).unwrap();
    assert!(
        src.contains("powf(10000.0"),
        "must compute frequency with base=10000.0"
    );
}

#[test]
fn test_rope_balanced_braces() {
    let config = PtxRopeConfig::new(512, 64);
    let src = generate_rope_ptx(&config).unwrap();
    let open = src.matches('{').count();
    let close = src.matches('}').count();
    assert_eq!(
        open, close,
        "braces must be balanced: {open} open, {close} close"
    );
}

// =========================================================================
// Cached kernel: structural validation
// =========================================================================

#[test]
fn test_cached_contains_kernel_name() {
    let config = PtxRopeConfig::new(512, 64);
    let src = generate_rope_cached_ptx(&config).unwrap();
    assert!(src.contains("rope_apply_cached"));
}

#[test]
fn test_cached_contains_cos_sin_table() {
    let config = PtxRopeConfig::new(512, 64);
    let src = generate_rope_cached_ptx(&config).unwrap();
    assert!(src.contains("cos_table"));
    assert!(src.contains("sin_table"));
}

#[test]
fn test_cached_no_trig_intrinsics() {
    let config = PtxRopeConfig::new(512, 64);
    let src = generate_rope_cached_ptx(&config).unwrap();
    assert!(
        !src.contains("__cosf"),
        "cached variant must NOT compute sin/cos"
    );
    assert!(
        !src.contains("__sinf"),
        "cached variant must NOT compute sin/cos"
    );
}

#[test]
fn test_cached_has_four_restrict_pointers() {
    let config = PtxRopeConfig::new(512, 64);
    let src = generate_rope_cached_ptx(&config).unwrap();
    assert_eq!(
        src.matches("__restrict__").count(),
        4,
        "must have 4 __restrict__ pointers (input, output, cos_table, sin_table)"
    );
}

#[test]
fn test_cached_header_documents_config() {
    let config = PtxRopeConfig::new(2048, 128);
    let src = generate_rope_cached_ptx(&config).unwrap();
    assert!(src.contains("seq_len=2048"));
    assert!(src.contains("head_dim=128"));
}

// =========================================================================
// Different configs produce different output
// =========================================================================

#[test]
fn test_different_seq_len_different_output() {
    let src_128 = generate_rope_ptx(&PtxRopeConfig::new(128, 64)).unwrap();
    let src_512 = generate_rope_ptx(&PtxRopeConfig::new(512, 64)).unwrap();
    assert_ne!(src_128, src_512);
}

#[test]
fn test_different_head_dim_different_output() {
    let src_64 = generate_rope_ptx(&PtxRopeConfig::new(512, 64)).unwrap();
    let src_128 = generate_rope_ptx(&PtxRopeConfig::new(512, 128)).unwrap();
    assert_ne!(src_64, src_128);
}

// =========================================================================
// Launch config
// =========================================================================

#[test]
fn test_launch_config_basic() {
    let config = PtxRopeConfig::new(512, 64);
    let (grid, block) = ptx_rope_launch_config(512, &config);
    // total_pairs = 512 * 32 = 16384, grid = ceil(16384 / 256) = 64
    assert_eq!(grid, 64);
    assert_eq!(block, 256);
}

#[test]
fn test_launch_config_single_position() {
    let config = PtxRopeConfig::new(1, 64);
    let (grid, block) = ptx_rope_launch_config(1, &config);
    // total_pairs = 1 * 32 = 32, grid = ceil(32 / 256) = 1
    assert_eq!(grid, 1);
    assert_eq!(block, 256);
}

#[test]
fn test_launch_config_small_head_dim() {
    let config = PtxRopeConfig::new(10, 4);
    let (grid, block) = ptx_rope_launch_config(10, &config);
    // total_pairs = 10 * 2 = 20, grid = ceil(20 / 256) = 1
    assert_eq!(grid, 1);
    assert_eq!(block, 256);
}

// =========================================================================
// CPU reference: rope_reference
// =========================================================================

#[test]
fn test_reference_zero_position() {
    // At position 0, theta = 0 for all dimensions.
    // cos(0) = 1, sin(0) = 0, so output should equal input.
    let head_dim = 4;
    let x = vec![1.0, 2.0, 3.0, 4.0]; // seq_len=1, head_dim=4
    let out = rope_reference(&x, 1, head_dim);
    for (o, &expected) in out.iter().zip(x.iter()) {
        assert!(
            (o - expected).abs() < 1e-6,
            "at pos=0, output should equal input: got {o}, expected {expected}"
        );
    }
}

#[test]
fn test_reference_known_angle_values() {
    // seq_len=2, head_dim=2: at position 1, one pair.
    // theta = 1 / 10000^(0/2) = 1.0
    // cos(1) ~ 0.5403, sin(1) ~ 0.8415
    let head_dim = 2;
    // Position 0: [0.0, 0.0] (padding), Position 1: [1.0, 0.0]
    let x = vec![0.0, 0.0, 1.0, 0.0];
    let out = rope_reference(&x, 2, head_dim);

    let theta = 1.0f32;
    let expected_0 = 1.0 * theta.cos() - 0.0 * theta.sin();
    let expected_1 = 1.0 * theta.sin() + 0.0 * theta.cos();

    // Check position 1 (offset 2..4)
    assert!(
        (out[2] - expected_0).abs() < 1e-5,
        "got {}, expected {}",
        out[2],
        expected_0
    );
    assert!(
        (out[3] - expected_1).abs() < 1e-5,
        "got {}, expected {}",
        out[3],
        expected_1
    );
}

#[test]
fn test_reference_dimension_pairing_correctness() {
    // Ensure pairs (0,1), (2,3), (4,5), etc. are handled independently.
    let head_dim = 6;
    let seq_len = 1;
    // Set only pair (2,3) to nonzero, others zero.
    let x = vec![0.0, 0.0, 5.0, 3.0, 0.0, 0.0];
    let out = rope_reference(&x, seq_len, head_dim);

    // At pos=0, theta=0 for all pairs, so output = input.
    assert!((out[0]).abs() < 1e-6);
    assert!((out[1]).abs() < 1e-6);
    assert!((out[2] - 5.0).abs() < 1e-6);
    assert!((out[3] - 3.0).abs() < 1e-6);
    assert!((out[4]).abs() < 1e-6);
    assert!((out[5]).abs() < 1e-6);
}

#[test]
fn test_reference_rotation_preserves_norm() {
    // A 2D rotation preserves the vector norm for each pair.
    let head_dim = 4;
    let seq_len = 3;
    let x: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i as f32 + 1.0) * 0.5)
        .collect();
    let out = rope_reference(&x, seq_len, head_dim);

    for pos in 0..seq_len {
        for pair in 0..(head_dim / 2) {
            let offset = pos * head_dim + 2 * pair;
            let in_norm = x[offset].hypot(x[offset + 1]);
            let out_norm = out[offset].hypot(out[offset + 1]);
            assert!(
                (in_norm - out_norm).abs() < 1e-5,
                "rotation must preserve norm: in={in_norm}, out={out_norm} at pos={pos}, pair={pair}"
            );
        }
    }
}

#[test]
fn test_reference_multiple_positions() {
    // Position 0 should be identity (theta=0).
    // Position 1 should differ from input.
    let head_dim = 4;
    let seq_len = 2;
    let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let out = rope_reference(&x, seq_len, head_dim);

    // Position 0: output should match input (theta=0 for all).
    assert!((out[0] - 1.0).abs() < 1e-6);
    assert!((out[1] - 0.0).abs() < 1e-6);

    // Position 1: output should differ from input (nonzero theta).
    let diff = (out[4] - 1.0).abs() + (out[5] - 0.0).abs();
    assert!(
        diff > 1e-4,
        "position 1 output should differ from input, diff={diff}"
    );
}

#[test]
fn test_reference_with_custom_base() {
    let head_dim = 4;
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // seq_len=2
    let out_default = rope_reference(&x, 2, head_dim);
    let out_custom = rope_reference_with_base(&x, 2, head_dim, 500.0);

    // Different bases should produce different outputs (for pos > 0).
    assert_ne!(
        out_default[4..8],
        out_custom[4..8],
        "different base should produce different results at pos=1"
    );
}

#[test]
fn test_reference_output_length() {
    let head_dim = 8;
    let seq_len = 5;
    let x = vec![0.0f32; seq_len * head_dim];
    let out = rope_reference(&x, seq_len, head_dim);
    assert_eq!(out.len(), seq_len * head_dim);
}

#[test]
#[should_panic(expected = "head_dim must be even")]
fn test_reference_odd_head_dim_panics() {
    let x = vec![0.0f32; 3];
    rope_reference(&x, 1, 3);
}

#[test]
#[should_panic(expected = "input length mismatch")]
fn test_reference_length_mismatch_panics() {
    let x = vec![0.0f32; 10];
    rope_reference(&x, 2, 4); // expects 8 elements, got 10
}

// =========================================================================
// ROPE_BLOCK_SIZE constant
// =========================================================================

#[test]
fn test_rope_block_size_constant() {
    assert_eq!(ROPE_BLOCK_SIZE, 256);
}

// =========================================================================
// Validation via generate rejects bad config
// =========================================================================

#[test]
fn test_generate_rejects_zero_seq_len() {
    let config = PtxRopeConfig::new(0, 64);
    assert!(generate_rope_ptx(&config).is_err());
}

#[test]
fn test_generate_rejects_odd_head_dim() {
    let config = PtxRopeConfig::new(512, 65);
    assert!(generate_rope_ptx(&config).is_err());
}

#[test]
fn test_generate_cached_rejects_bad_config() {
    let config = PtxRopeConfig::new(512, 63);
    assert!(generate_rope_cached_ptx(&config).is_err());
}
