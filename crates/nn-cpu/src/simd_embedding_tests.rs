// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for CPU SIMD embedding lookup implementation.

use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a deterministic weight table [vocab_size, embed_dim] where
/// `weights[v][d] = (v * embed_dim + d) as f32 * 0.01`.
fn make_weights(vocab_size: usize, embed_dim: usize) -> Vec<f32> {
    (0..vocab_size * embed_dim)
        .map(|i| i as f32 * 0.01)
        .collect()
}

/// Extract row `idx` from a flat weight table.
fn get_row(weights: &[f32], idx: usize, embed_dim: usize) -> &[f32] {
    let start = idx * embed_dim;
    &weights[start..start + embed_dim]
}

// ---------------------------------------------------------------------------
// Basic single-index lookup
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_single_index_first_row() {
    let embed_dim = 4;
    let weights = make_weights(10, embed_dim);
    let indices = [0u32];
    let mut output = vec![0.0f32; embed_dim];

    embedding_scalar(&weights, &indices, &mut output, embed_dim)
        .expect("single index lookup should succeed");

    let expected = get_row(&weights, 0, embed_dim);
    assert_eq!(output, expected, "first row lookup mismatch");
}

#[test]
fn test_scalar_single_index_last_row() {
    let embed_dim = 4;
    let vocab_size = 10;
    let weights = make_weights(vocab_size, embed_dim);
    let indices = [(vocab_size - 1) as u32];
    let mut output = vec![0.0f32; embed_dim];

    embedding_scalar(&weights, &indices, &mut output, embed_dim)
        .expect("last row lookup should succeed");

    let expected = get_row(&weights, vocab_size - 1, embed_dim);
    assert_eq!(output, expected, "last row lookup mismatch");
}

#[test]
fn test_scalar_single_index_middle_row() {
    let embed_dim = 8;
    let vocab_size = 100;
    let weights = make_weights(vocab_size, embed_dim);
    let indices = [42u32];
    let mut output = vec![0.0f32; embed_dim];

    embedding_scalar(&weights, &indices, &mut output, embed_dim)
        .expect("middle row lookup should succeed");

    let expected = get_row(&weights, 42, embed_dim);
    assert_eq!(output, expected, "middle row lookup mismatch");
}

#[test]
fn test_dispatch_single_index() {
    let embed_dim = 16;
    let weights = make_weights(50, embed_dim);
    let indices = [7u32];
    let mut output = vec![0.0f32; embed_dim];

    embedding(&weights, &indices, &mut output, embed_dim)
        .expect("dispatch single index should succeed");

    let expected = get_row(&weights, 7, embed_dim);
    assert_eq!(output, expected, "dispatch single index mismatch");
}

// ---------------------------------------------------------------------------
// Batched lookup with multiple indices
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_batched_lookup() {
    let embed_dim = 4;
    let vocab_size = 10;
    let weights = make_weights(vocab_size, embed_dim);
    let indices = [0u32, 5, 9, 3];
    let batch = indices.len();
    let mut output = vec![0.0f32; batch * embed_dim];

    embedding_scalar(&weights, &indices, &mut output, embed_dim)
        .expect("batched lookup should succeed");

    for (b, &idx) in indices.iter().enumerate() {
        let expected = get_row(&weights, idx as usize, embed_dim);
        let actual = &output[b * embed_dim..(b + 1) * embed_dim];
        assert_eq!(actual, expected, "batch {b} (idx={idx}) mismatch");
    }
}

#[test]
fn test_dispatch_batched_lookup() {
    let embed_dim = 32;
    let vocab_size = 100;
    let weights = make_weights(vocab_size, embed_dim);
    let indices = [0u32, 1, 50, 99, 25, 75, 10, 90];
    let batch = indices.len();
    let mut output = vec![0.0f32; batch * embed_dim];

    embedding(&weights, &indices, &mut output, embed_dim)
        .expect("dispatch batched lookup should succeed");

    for (b, &idx) in indices.iter().enumerate() {
        let expected = get_row(&weights, idx as usize, embed_dim);
        let actual = &output[b * embed_dim..(b + 1) * embed_dim];
        assert_eq!(actual, expected, "dispatch batch {b} (idx={idx}) mismatch");
    }
}

#[test]
fn test_batched_lookup_duplicate_indices() {
    let embed_dim = 8;
    let weights = make_weights(10, embed_dim);
    let indices = [3u32, 3, 3, 7, 7];
    let batch = indices.len();
    let mut output = vec![0.0f32; batch * embed_dim];

    embedding(&weights, &indices, &mut output, embed_dim)
        .expect("duplicate indices lookup should succeed");

    // First three should all be row 3, last two row 7.
    let row3 = get_row(&weights, 3, embed_dim);
    let row7 = get_row(&weights, 7, embed_dim);
    for b in 0..3 {
        assert_eq!(
            &output[b * embed_dim..(b + 1) * embed_dim],
            row3,
            "duplicate idx=3, batch {b}"
        );
    }
    for b in 3..5 {
        assert_eq!(
            &output[b * embed_dim..(b + 1) * embed_dim],
            row7,
            "duplicate idx=7, batch {b}"
        );
    }
}

#[test]
fn test_batched_lookup_sequential_indices() {
    let embed_dim = 16;
    let vocab_size = 20;
    let weights = make_weights(vocab_size, embed_dim);
    let indices: Vec<u32> = (0..vocab_size as u32).collect();
    let batch = indices.len();
    let mut output = vec![0.0f32; batch * embed_dim];

    embedding(&weights, &indices, &mut output, embed_dim)
        .expect("sequential lookup should succeed");

    // Output should be identical to weights (all rows in order).
    assert_eq!(
        output, weights,
        "sequential lookup should reproduce weights"
    );
}

// ---------------------------------------------------------------------------
// Out-of-bounds index handling
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_oob_index_returns_error() {
    let embed_dim = 4;
    let vocab_size = 10;
    let weights = make_weights(vocab_size, embed_dim);
    let indices = [10u32]; // vocab_size is 10, so index 10 is OOB.
    let mut output = vec![0.0f32; embed_dim];

    let result = embedding_scalar(&weights, &indices, &mut output, embed_dim);
    match result {
        Err(EmbeddingError::IndexOutOfBounds {
            index: 10,
            vocab_size: 10,
            batch_position: 0,
        }) => {} // expected
        other => panic!("expected IndexOutOfBounds, got {other:?}"),
    }
}

#[test]
fn test_dispatch_oob_index_returns_error() {
    let embed_dim = 8;
    let vocab_size = 5;
    let weights = make_weights(vocab_size, embed_dim);
    let indices = [2u32, 5]; // index 5 is OOB for vocab_size 5.
    let mut output = vec![0.0f32; 2 * embed_dim];

    let result = embedding(&weights, &indices, &mut output, embed_dim);
    match result {
        Err(EmbeddingError::IndexOutOfBounds {
            index: 5,
            vocab_size: 5,
            batch_position: 1,
        }) => {} // expected
        other => panic!("expected IndexOutOfBounds at position 1, got {other:?}"),
    }
}

#[test]
fn test_oob_large_index() {
    let embed_dim = 4;
    let weights = make_weights(10, embed_dim);
    let indices = [u32::MAX];
    let mut output = vec![0.0f32; embed_dim];

    let result = embedding(&weights, &indices, &mut output, embed_dim);
    assert!(
        matches!(result, Err(EmbeddingError::IndexOutOfBounds { .. })),
        "u32::MAX should be OOB"
    );
}

#[test]
fn test_oob_no_partial_writes() {
    // When an index is OOB, the output should not be partially written.
    // (Fail-fast validation means no work is done before returning error.)
    let embed_dim = 4;
    let weights = make_weights(10, embed_dim);
    let indices = [0u32, 100]; // index 100 is OOB.
    let mut output = vec![f32::NAN; 2 * embed_dim];

    let result = embedding_scalar(&weights, &indices, &mut output, embed_dim);
    assert!(result.is_err(), "should fail on OOB index");
    // Output should be untouched (all NaN) because validation is fail-fast.
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_nan(), "output[{i}] was modified despite error");
    }
}

// ---------------------------------------------------------------------------
// Invalid weight shape / output length
// ---------------------------------------------------------------------------

#[test]
fn test_invalid_weight_shape() {
    let embed_dim = 4;
    // 11 is not a multiple of 4.
    let weights = vec![0.0f32; 11];
    let indices = [0u32];
    let mut output = vec![0.0f32; embed_dim];

    let result = embedding(&weights, &indices, &mut output, embed_dim);
    assert!(
        matches!(result, Err(EmbeddingError::InvalidWeightShape { .. })),
        "should reject non-multiple weight length"
    );
}

#[test]
fn test_output_length_mismatch() {
    let embed_dim = 4;
    let weights = make_weights(10, embed_dim);
    let indices = [0u32, 1];
    // Output should be 2 * 4 = 8, but we provide 4.
    let mut output = vec![0.0f32; embed_dim];

    let result = embedding(&weights, &indices, &mut output, embed_dim);
    assert!(
        matches!(result, Err(EmbeddingError::OutputLengthMismatch { .. })),
        "should reject mismatched output length"
    );
}

#[test]
fn test_zero_embed_dim() {
    let weights: Vec<f32> = vec![];
    let indices = [0u32];
    let mut output: Vec<f32> = vec![];

    let result = embedding(&weights, &indices, &mut output, 0);
    assert!(
        matches!(result, Err(EmbeddingError::ZeroEmbedDim)),
        "should reject zero embed_dim"
    );
}

// ---------------------------------------------------------------------------
// Known-value computation verification
// ---------------------------------------------------------------------------

#[test]
fn test_known_values_small() {
    // Manually specified weight table and expected outputs.
    let embed_dim = 3;
    #[rustfmt::skip]
    let weights = vec![
        1.0, 2.0, 3.0,   // row 0
        4.0, 5.0, 6.0,   // row 1
        7.0, 8.0, 9.0,   // row 2
        10.0, 11.0, 12.0, // row 3
    ];
    let indices = [2u32, 0, 3, 1];
    let mut output = vec![0.0f32; 4 * embed_dim];

    embedding(&weights, &indices, &mut output, embed_dim)
        .expect("known-value lookup should succeed");

    let expected = vec![
        7.0, 8.0, 9.0, // row 2
        1.0, 2.0, 3.0, // row 0
        10.0, 11.0, 12.0, // row 3
        4.0, 5.0, 6.0, // row 1
    ];
    assert_eq!(output, expected, "known-value mismatch");
}

#[test]
fn test_known_values_single_vocab() {
    // vocab_size=1: any index 0 should return the only row.
    let embed_dim = 5;
    let weights = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let indices = [0u32, 0, 0];
    let mut output = vec![0.0f32; 3 * embed_dim];

    embedding(&weights, &indices, &mut output, embed_dim)
        .expect("single-vocab lookup should succeed");

    for b in 0..3 {
        assert_eq!(
            &output[b * embed_dim..(b + 1) * embed_dim],
            &weights[..],
            "single vocab batch {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Various embedding dimensions
// ---------------------------------------------------------------------------

#[test]
fn test_embed_dim_64() {
    let embed_dim = 64;
    let vocab_size = 50;
    let weights = make_weights(vocab_size, embed_dim);
    let indices = [0u32, 25, 49];
    let result =
        embedding_lookup(&weights, &indices, embed_dim).expect("embed_dim=64 should succeed");
    for (b, &idx) in indices.iter().enumerate() {
        let expected = get_row(&weights, idx as usize, embed_dim);
        assert_eq!(
            &result[b * embed_dim..(b + 1) * embed_dim],
            expected,
            "embed_dim=64, batch {b}"
        );
    }
}

#[test]
fn test_embed_dim_128() {
    let embed_dim = 128;
    let vocab_size = 30;
    let weights = make_weights(vocab_size, embed_dim);
    let indices = [5u32, 15, 29];
    let result =
        embedding_lookup(&weights, &indices, embed_dim).expect("embed_dim=128 should succeed");
    for (b, &idx) in indices.iter().enumerate() {
        let expected = get_row(&weights, idx as usize, embed_dim);
        assert_eq!(
            &result[b * embed_dim..(b + 1) * embed_dim],
            expected,
            "embed_dim=128, batch {b}"
        );
    }
}

#[test]
fn test_embed_dim_256() {
    let embed_dim = 256;
    let vocab_size = 20;
    let weights = make_weights(vocab_size, embed_dim);
    let indices = [0u32, 10, 19];
    let result =
        embedding_lookup(&weights, &indices, embed_dim).expect("embed_dim=256 should succeed");
    for (b, &idx) in indices.iter().enumerate() {
        let expected = get_row(&weights, idx as usize, embed_dim);
        assert_eq!(
            &result[b * embed_dim..(b + 1) * embed_dim],
            expected,
            "embed_dim=256, batch {b}"
        );
    }
}

#[test]
fn test_embed_dim_512() {
    let embed_dim = 512;
    let vocab_size = 15;
    let weights = make_weights(vocab_size, embed_dim);
    let indices = [3u32, 7, 14];
    let result =
        embedding_lookup(&weights, &indices, embed_dim).expect("embed_dim=512 should succeed");
    for (b, &idx) in indices.iter().enumerate() {
        let expected = get_row(&weights, idx as usize, embed_dim);
        assert_eq!(
            &result[b * embed_dim..(b + 1) * embed_dim],
            expected,
            "embed_dim=512, batch {b}"
        );
    }
}

#[test]
fn test_embed_dim_768() {
    let embed_dim = 768;
    let vocab_size = 10;
    let weights = make_weights(vocab_size, embed_dim);
    let indices = [0u32, 5, 9];
    let result =
        embedding_lookup(&weights, &indices, embed_dim).expect("embed_dim=768 should succeed");
    for (b, &idx) in indices.iter().enumerate() {
        let expected = get_row(&weights, idx as usize, embed_dim);
        assert_eq!(
            &result[b * embed_dim..(b + 1) * embed_dim],
            expected,
            "embed_dim=768, batch {b}"
        );
    }
}

#[test]
fn test_embed_dim_1024() {
    let embed_dim = 1024;
    let vocab_size = 8;
    let weights = make_weights(vocab_size, embed_dim);
    let indices = [0u32, 4, 7];
    let result =
        embedding_lookup(&weights, &indices, embed_dim).expect("embed_dim=1024 should succeed");
    for (b, &idx) in indices.iter().enumerate() {
        let expected = get_row(&weights, idx as usize, embed_dim);
        assert_eq!(
            &result[b * embed_dim..(b + 1) * embed_dim],
            expected,
            "embed_dim=1024, batch {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Various vocab sizes
// ---------------------------------------------------------------------------

#[test]
fn test_vocab_size_1() {
    let embed_dim = 16;
    let weights = make_weights(1, embed_dim);
    let indices = [0u32];
    let result =
        embedding_lookup(&weights, &indices, embed_dim).expect("vocab_size=1 should succeed");
    assert_eq!(result, weights);
}

#[test]
fn test_vocab_size_large() {
    let embed_dim = 32;
    let vocab_size = 50000;
    let weights = make_weights(vocab_size, embed_dim);
    let indices = [0u32, 1000, 25000, 49999];
    let result =
        embedding_lookup(&weights, &indices, embed_dim).expect("large vocab should succeed");
    for (b, &idx) in indices.iter().enumerate() {
        let expected = get_row(&weights, idx as usize, embed_dim);
        assert_eq!(
            &result[b * embed_dim..(b + 1) * embed_dim],
            expected,
            "large vocab batch {b} (idx={idx})"
        );
    }
}

// ---------------------------------------------------------------------------
// Non-multiple-of-SIMD-width embed_dim (tests scalar tail)
// ---------------------------------------------------------------------------

#[test]
fn test_embed_dim_1() {
    let embed_dim = 1;
    let weights = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let indices = [2u32, 4, 0];
    let result =
        embedding_lookup(&weights, &indices, embed_dim).expect("embed_dim=1 should succeed");
    assert_eq!(result, vec![30.0, 50.0, 10.0]);
}

#[test]
fn test_embed_dim_3() {
    // 3 is not a multiple of 4 (NEON) or 8 (AVX2).
    let embed_dim = 3;
    let weights = make_weights(5, embed_dim);
    let indices = [0u32, 1, 2, 3, 4];
    let result =
        embedding_lookup(&weights, &indices, embed_dim).expect("embed_dim=3 should succeed");
    // Should reproduce the full weight table.
    assert_eq!(result, weights);
}

#[test]
fn test_embed_dim_5() {
    let embed_dim = 5;
    let weights = make_weights(10, embed_dim);
    let indices = [9u32, 0];
    let result =
        embedding_lookup(&weights, &indices, embed_dim).expect("embed_dim=5 should succeed");
    let expected_row9 = get_row(&weights, 9, embed_dim);
    let expected_row0 = get_row(&weights, 0, embed_dim);
    assert_eq!(&result[..embed_dim], expected_row9);
    assert_eq!(&result[embed_dim..], expected_row0);
}

#[test]
fn test_embed_dim_7() {
    // 7 = 1 NEON chunk + 3 tail, or 0 AVX2 chunks + 7 tail.
    let embed_dim = 7;
    let weights = make_weights(8, embed_dim);
    let indices = [3u32, 5, 7];
    let result =
        embedding_lookup(&weights, &indices, embed_dim).expect("embed_dim=7 should succeed");
    for (b, &idx) in indices.iter().enumerate() {
        let expected = get_row(&weights, idx as usize, embed_dim);
        assert_eq!(
            &result[b * embed_dim..(b + 1) * embed_dim],
            expected,
            "embed_dim=7, batch {b}"
        );
    }
}

#[test]
fn test_embed_dim_9() {
    // 9 = 2 NEON chunks + 1 tail, or 1 AVX2 chunk + 1 tail.
    let embed_dim = 9;
    let weights = make_weights(6, embed_dim);
    let indices = [0u32, 3, 5];
    let result =
        embedding_lookup(&weights, &indices, embed_dim).expect("embed_dim=9 should succeed");
    for (b, &idx) in indices.iter().enumerate() {
        let expected = get_row(&weights, idx as usize, embed_dim);
        assert_eq!(
            &result[b * embed_dim..(b + 1) * embed_dim],
            expected,
            "embed_dim=9, batch {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// SIMD vs scalar path agreement
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_matches_scalar_varied_dims() {
    // Test multiple dimensions to exercise SIMD tail handling.
    for embed_dim in [
        1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 256,
    ] {
        let vocab_size = 20;
        let weights = make_weights(vocab_size, embed_dim);
        let indices: Vec<u32> = (0..vocab_size as u32).collect();
        let batch = indices.len();

        let mut scalar_out = vec![0.0f32; batch * embed_dim];
        embedding_scalar(&weights, &indices, &mut scalar_out, embed_dim)
            .expect("scalar should succeed");

        let mut dispatch_out = vec![0.0f32; batch * embed_dim];
        embedding(&weights, &indices, &mut dispatch_out, embed_dim)
            .expect("dispatch should succeed");

        assert_eq!(
            scalar_out, dispatch_out,
            "dispatch != scalar for embed_dim={embed_dim}"
        );
    }
}

#[test]
fn test_dispatch_matches_reference_varied_dims() {
    for embed_dim in [4, 8, 16, 32, 64, 128, 256, 512, 768] {
        let vocab_size = 10;
        let weights = make_weights(vocab_size, embed_dim);
        let indices = [0u32, 5, 9];

        let reference =
            embedding_reference(&weights, &indices, embed_dim).expect("reference should succeed");

        let dispatch =
            embedding_lookup(&weights, &indices, embed_dim).expect("dispatch should succeed");

        assert_eq!(
            reference, dispatch,
            "dispatch != reference for embed_dim={embed_dim}"
        );
    }
}

// ---------------------------------------------------------------------------
// Empty batch (zero indices)
// ---------------------------------------------------------------------------

#[test]
fn test_empty_batch() {
    let embed_dim = 8;
    let weights = make_weights(10, embed_dim);
    let indices: &[u32] = &[];
    let mut output: Vec<f32> = vec![];

    embedding(&weights, indices, &mut output, embed_dim).expect("empty batch should succeed");
    assert!(output.is_empty(), "empty batch should produce empty output");
}

// ---------------------------------------------------------------------------
// embedding_reference convenience function
// ---------------------------------------------------------------------------

#[test]
fn test_reference_returns_correct_allocation() {
    let embed_dim = 16;
    let vocab_size = 10;
    let weights = make_weights(vocab_size, embed_dim);
    let indices = [3u32, 7, 1];

    let result =
        embedding_reference(&weights, &indices, embed_dim).expect("reference should succeed");

    assert_eq!(result.len(), 3 * embed_dim, "output length mismatch");
    for (b, &idx) in indices.iter().enumerate() {
        let expected = get_row(&weights, idx as usize, embed_dim);
        assert_eq!(
            &result[b * embed_dim..(b + 1) * embed_dim],
            expected,
            "reference batch {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// embedding_lookup convenience function
// ---------------------------------------------------------------------------

#[test]
fn test_lookup_convenience_function() {
    let embed_dim = 32;
    let vocab_size = 50;
    let weights = make_weights(vocab_size, embed_dim);
    let indices = [10u32, 20, 30, 40];

    let result = embedding_lookup(&weights, &indices, embed_dim).expect("lookup should succeed");

    assert_eq!(result.len(), 4 * embed_dim);
    for (b, &idx) in indices.iter().enumerate() {
        let expected = get_row(&weights, idx as usize, embed_dim);
        assert_eq!(
            &result[b * embed_dim..(b + 1) * embed_dim],
            expected,
            "lookup batch {b}"
        );
    }
}

#[test]
fn test_lookup_zero_embed_dim_error() {
    let weights: Vec<f32> = vec![];
    let indices = [0u32];
    let result = embedding_lookup(&weights, &indices, 0);
    assert!(
        matches!(result, Err(EmbeddingError::ZeroEmbedDim)),
        "lookup with zero embed_dim should error"
    );
}

// ---------------------------------------------------------------------------
// SIMD path selection (verify the dispatch works)
// ---------------------------------------------------------------------------

#[test]
fn test_simd_detection_available() {
    use crate::simd_detect;
    let level = simd_detect::detect();
    // On aarch64 this should be Neon, on x86_64 Avx2 or Scalar.
    #[cfg(target_arch = "aarch64")]
    assert_eq!(level, simd_detect::SimdLevel::Neon);
    #[cfg(target_arch = "x86_64")]
    assert!(
        level == simd_detect::SimdLevel::Avx2 || level == simd_detect::SimdLevel::Scalar,
        "x86_64 should detect Avx2 or Scalar"
    );
}

#[test]
fn test_large_batch_large_dim() {
    // Stress test: 1000 indices into a 10k vocab with embed_dim=512.
    let embed_dim = 512;
    let vocab_size = 10_000;
    let weights = make_weights(vocab_size, embed_dim);
    let indices: Vec<u32> = (0..1000).map(|i| (i * 7) % vocab_size as u32).collect();

    let result =
        embedding_lookup(&weights, &indices, embed_dim).expect("large batch+dim should succeed");

    assert_eq!(result.len(), 1000 * embed_dim);
    // Spot-check a few entries.
    for &b in &[0, 100, 500, 999] {
        let idx = indices[b] as usize;
        let expected = get_row(&weights, idx, embed_dim);
        assert_eq!(
            &result[b * embed_dim..(b + 1) * embed_dim],
            expected,
            "large batch spot-check at {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Error display formatting
// ---------------------------------------------------------------------------

#[test]
fn test_error_display() {
    let e = EmbeddingError::IndexOutOfBounds {
        index: 42,
        vocab_size: 30,
        batch_position: 5,
    };
    let msg = format!("{e}");
    assert!(msg.contains("42"), "should contain index");
    assert!(msg.contains("30"), "should contain vocab_size");
    assert!(msg.contains("5"), "should contain batch_position");

    let e2 = EmbeddingError::ZeroEmbedDim;
    let msg2 = format!("{e2}");
    assert!(msg2.contains("embed_dim"), "should mention embed_dim");
}
