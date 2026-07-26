#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for neural codec token algebra.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

use super::*;

/// Helper: create a simple codebook with known values for testing.
/// Level 0: entries are [0, 0, ...], [1, 1, ...], [2, 2, ...], etc.
/// Level 1: entries are [0.1, 0.1, ...], [0.2, 0.2, ...], etc.
fn make_test_codebooks(vocab_size: usize, embed_dim: usize, n_levels: usize) -> Vec<DynTensor> {
    let device = Device::Cpu;
    let mut codebooks = Vec::new();

    for level in 0..n_levels {
        let scale = if level == 0 {
            1.0
        } else {
            0.1_f32.powi(level as i32)
        };
        let mut data = Vec::with_capacity(vocab_size * embed_dim);
        for entry in 0..vocab_size {
            for _dim in 0..embed_dim {
                data.push(entry as f32 * scale);
            }
        }
        let tensor =
            DynTensor::new(&data, &[vocab_size, embed_dim], &device).expect("valid codebook");
        codebooks.push(tensor);
    }

    codebooks
}

#[test]
fn test_from_codebooks_valid() {
    let codebooks = make_test_codebooks(16, 8, 2);
    let space = CodecEmbeddingSpace::from_codebooks(codebooks).unwrap();
    assert_eq!(space.n_levels(), 2);
    assert_eq!(space.embed_dim(), 8);
    assert_eq!(space.vocab_size(), 16);
}

#[test]
fn test_from_codebooks_empty() {
    let result = CodecEmbeddingSpace::from_codebooks(vec![]);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("at least one codebook"), "got: {err}");
}

#[test]
fn test_from_codebooks_shape_mismatch() {
    let device = Device::Cpu;
    let cb1 = DynTensor::zeros(&[16, 8], DType::F32, &device).unwrap();
    let cb2 = DynTensor::zeros(&[16, 4], DType::F32, &device).unwrap();
    let result = CodecEmbeddingSpace::from_codebooks(vec![cb1, cb2]);
    assert!(result.is_err());
}

#[test]
fn test_embed_single_level() {
    let codebooks = make_test_codebooks(16, 4, 1);
    let space = CodecEmbeddingSpace::from_codebooks(codebooks).unwrap();

    // Token 3 at level 0 → embedding [3.0, 3.0, 3.0, 3.0]
    let tokens = vec![vec![3]];
    let emb = space.embed(&tokens).unwrap();
    assert_eq!(emb.dims(), &[1, 4]);

    let values = emb.to_flat_vec::<f32>().unwrap();
    assert!((values[0] - 3.0).abs() < 1e-6);
    assert!((values[3] - 3.0).abs() < 1e-6);
}

#[test]
fn test_embed_residual_sum() {
    let codebooks = make_test_codebooks(16, 4, 2);
    let space = CodecEmbeddingSpace::from_codebooks(codebooks).unwrap();

    // Level 0 token 3: [3.0, 3.0, 3.0, 3.0]
    // Level 1 token 5: [0.5, 0.5, 0.5, 0.5]
    // Sum: [3.5, 3.5, 3.5, 3.5]
    let tokens = vec![vec![3], vec![5]];
    let emb = space.embed(&tokens).unwrap();

    let values = emb.to_flat_vec::<f32>().unwrap();
    assert!((values[0] - 3.5).abs() < 1e-5, "got {}", values[0]);
}

#[test]
fn test_embed_wrong_levels() {
    let codebooks = make_test_codebooks(16, 4, 2);
    let space = CodecEmbeddingSpace::from_codebooks(codebooks).unwrap();

    // Only 1 level provided when 2 expected
    let tokens = vec![vec![3]];
    let result = space.embed(&tokens);
    assert!(result.is_err());
}

#[test]
fn test_embed_out_of_range_token() {
    let codebooks = make_test_codebooks(16, 4, 1);
    let space = CodecEmbeddingSpace::from_codebooks(codebooks).unwrap();

    let tokens = vec![vec![99]]; // 99 >= 16
    let result = space.embed(&tokens);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("vocab_size"), "got: {err}");
}

#[test]
fn test_analogy_basic() {
    let codebooks = make_test_codebooks(16, 4, 1);
    let space = CodecEmbeddingSpace::from_codebooks(codebooks).unwrap();

    // a = embed(5) = [5, 5, 5, 5]
    // b = embed(2) = [2, 2, 2, 2]
    // c = embed(1) = [1, 1, 1, 1]
    // analogy = a - b + c = [4, 4, 4, 4]
    let a = space.embed(&[vec![5]]).unwrap();
    let b = space.embed(&[vec![2]]).unwrap();
    let c = space.embed(&[vec![1]]).unwrap();

    let result = space.analogy(&a, &b, &c).unwrap();
    let values = result.to_flat_vec::<f32>().unwrap();
    assert!((values[0] - 4.0).abs() < 1e-6, "got {}", values[0]);
}

#[test]
fn test_interpolation_endpoints() {
    let codebooks = make_test_codebooks(16, 4, 1);
    let space = CodecEmbeddingSpace::from_codebooks(codebooks).unwrap();

    let a = space.embed(&[vec![2]]).unwrap(); // [2, 2, 2, 2]
    let b = space.embed(&[vec![8]]).unwrap(); // [8, 8, 8, 8]

    // alpha=0 → pure a
    let r0 = space.interpolate(&a, &b, 0.0).unwrap();
    let v0 = r0.to_flat_vec::<f32>().unwrap();
    assert!((v0[0] - 2.0).abs() < 1e-6);

    // alpha=1 → pure b
    let r1 = space.interpolate(&a, &b, 1.0).unwrap();
    let v1 = r1.to_flat_vec::<f32>().unwrap();
    assert!((v1[0] - 8.0).abs() < 1e-6);

    // alpha=0.5 → midpoint
    let r5 = space.interpolate(&a, &b, 0.5).unwrap();
    let v5 = r5.to_flat_vec::<f32>().unwrap();
    assert!((v5[0] - 5.0).abs() < 1e-6);
}

#[test]
fn test_interpolation_invalid_alpha() {
    let codebooks = make_test_codebooks(16, 4, 1);
    let space = CodecEmbeddingSpace::from_codebooks(codebooks).unwrap();

    let a = space.embed(&[vec![0]]).unwrap();
    let b = space.embed(&[vec![1]]).unwrap();

    assert!(space.interpolate(&a, &b, -0.1).is_err());
    assert!(space.interpolate(&a, &b, 1.1).is_err());
    assert!(space.interpolate(&a, &b, f32::NAN).is_err());
}

#[test]
fn test_quantize_round_trip() {
    let codebooks = make_test_codebooks(16, 4, 1);
    let space = CodecEmbeddingSpace::from_codebooks(codebooks).unwrap();

    // Embed tokens, then quantize back — should recover original tokens
    let original_tokens = vec![vec![3, 7, 11]];
    let emb = space.embed(&original_tokens).unwrap();
    let recovered = space.quantize(&emb).unwrap();

    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0], vec![3, 7, 11]);
}

#[test]
fn test_quantize_round_trip_multi_level() {
    let codebooks = make_test_codebooks(16, 4, 2);
    let space = CodecEmbeddingSpace::from_codebooks(codebooks).unwrap();

    let original_tokens = vec![vec![3, 7], vec![5, 2]];
    let emb = space.embed(&original_tokens).unwrap();
    let recovered = space.quantize(&emb).unwrap();

    assert_eq!(recovered.len(), 2);
    assert_eq!(recovered[0], vec![3, 7], "level 0 mismatch");
    assert_eq!(recovered[1], vec![5, 2], "level 1 mismatch");
}

#[test]
fn test_utterance_centroid_single() {
    let codebooks = make_test_codebooks(16, 4, 1);
    let space = CodecEmbeddingSpace::from_codebooks(codebooks).unwrap();

    // Single utterance with tokens [2, 4, 6]
    // Embeddings: [2,2,2,2], [4,4,4,4], [6,6,6,6]
    // Mean: [4, 4, 4, 4]
    let utterances = vec![vec![vec![2, 4, 6]]];
    let centroid = utterance_centroid(&space, &utterances).unwrap();

    let values = centroid.to_flat_vec::<f32>().unwrap();
    assert!((values[0] - 4.0).abs() < 1e-6, "got {}", values[0]);
}

#[test]
fn test_utterance_centroid_multiple() {
    let codebooks = make_test_codebooks(16, 4, 1);
    let space = CodecEmbeddingSpace::from_codebooks(codebooks).unwrap();

    // Utterance 1: tokens [2] → embedding [2, 2, 2, 2] (1 frame)
    // Utterance 2: tokens [6] → embedding [6, 6, 6, 6] (1 frame)
    // Mean across 2 frames: [4, 4, 4, 4]
    let utterances = vec![vec![vec![2]], vec![vec![6]]];
    let centroid = utterance_centroid(&space, &utterances).unwrap();

    let values = centroid.to_flat_vec::<f32>().unwrap();
    assert!((values[0] - 4.0).abs() < 1e-6, "got {}", values[0]);
}

#[test]
fn test_utterance_centroid_empty() {
    let codebooks = make_test_codebooks(16, 4, 1);
    let space = CodecEmbeddingSpace::from_codebooks(codebooks).unwrap();

    let result = utterance_centroid(&space, &[]);
    assert!(result.is_err());
}

#[test]
fn test_voice_conversion_pattern() {
    // Voice conversion: utterance_A - speaker_A_centroid + speaker_B_centroid
    let codebooks = make_test_codebooks(16, 4, 1);
    let space = CodecEmbeddingSpace::from_codebooks(codebooks).unwrap();

    // Speaker A utterances: tokens [2, 4] → centroid = [3, 3, 3, 3]
    let speaker_a_utts = vec![vec![vec![2, 4]]];
    let centroid_a = speaker_centroid(&space, &speaker_a_utts).unwrap();

    // Speaker B utterances: tokens [8, 10] → centroid = [9, 9, 9, 9]
    let speaker_b_utts = vec![vec![vec![8, 10]]];
    let centroid_b = speaker_centroid(&space, &speaker_b_utts).unwrap();

    // Target utterance from speaker A: tokens [3]
    let utt_a = space.embed(&[vec![3]]).unwrap();

    // Voice conversion: utt_a - centroid_a + centroid_b
    // = [3,3,3,3] - [3,3,3,3] + [9,9,9,9] = [9,9,9,9]
    let converted = space.analogy(&utt_a, &centroid_a, &centroid_b).unwrap();
    let values = converted.to_flat_vec::<f32>().unwrap();
    assert!((values[0] - 9.0).abs() < 1e-5, "got {}", values[0]);
}

#[test]
fn test_quantize_wrong_shape() {
    let codebooks = make_test_codebooks(16, 4, 1);
    let space = CodecEmbeddingSpace::from_codebooks(codebooks).unwrap();

    // Wrong embed_dim
    let device = Device::Cpu;
    let wrong = DynTensor::zeros(&[3, 8], DType::F32, &device).unwrap();
    assert!(space.quantize(&wrong).is_err());

    // Wrong rank
    let wrong_rank = DynTensor::zeros(&[3], DType::F32, &device).unwrap();
    assert!(space.quantize(&wrong_rank).is_err());
}

#[test]
fn test_interpolation_quality_sweep() {
    // Verify interpolation produces smooth transition
    let codebooks = make_test_codebooks(16, 4, 1);
    let space = CodecEmbeddingSpace::from_codebooks(codebooks).unwrap();

    let a = space.embed(&[vec![2]]).unwrap(); // [2, 2, 2, 2]
    let b = space.embed(&[vec![10]]).unwrap(); // [10, 10, 10, 10]

    let alphas = [0.0, 0.25, 0.5, 0.75, 1.0];
    let mut prev_val = f32::NEG_INFINITY;

    for &alpha in &alphas {
        let result = space.interpolate(&a, &b, alpha).unwrap();
        let val = result.to_flat_vec::<f32>().unwrap()[0];
        // Values should monotonically increase from 2 to 10
        assert!(
            val > prev_val,
            "non-monotonic at alpha={alpha}: {val} <= {prev_val}"
        );
        prev_val = val;
    }

    // Verify endpoints
    let first = space
        .interpolate(&a, &b, 0.0)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()[0];
    let last = space
        .interpolate(&a, &b, 1.0)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()[0];
    assert!((first - 2.0).abs() < 1e-6);
    assert!((last - 10.0).abs() < 1e-6);
}
