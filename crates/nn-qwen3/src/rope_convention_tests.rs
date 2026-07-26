// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for RoPE convention correctness (#4327).
//!
//! Verifies that nn-qwen3's half-split RoPE (`apply_pair_half_split`)
//! matches HuggingFace's `rotate_half` convention for direct HF weights.
//!
//! ## Background
//!
//! There are two equivalent RoPE conventions:
//!
//! 1. **Interleaved** (pairs `(2i, 2i+1)`):
//!    ```text
//!    y[2i]   = x[2i]   * cos[i] - x[2i+1] * sin[i]
//!    y[2i+1] = x[2i]   * sin[i] + x[2i+1] * cos[i]
//!    ```
//!
//! 2. **Half-split** (pairs `(i, i+half_dim)`, HuggingFace `rotate_half`):
//!    ```text
//!    x1 = x[..., :half], x2 = x[..., half:]
//!    y1 = x1 * cos - x2 * sin
//!    y2 = x1 * sin + x2 * cos
//!    y  = cat(y1, y2)
//!    ```
//!
//! Both are mathematically equivalent IF Q/K weight columns are permuted to
//! match the convention. nn-qwen3 uses half-split (convention 2), which
//! matches HuggingFace directly -- NO weight permutation is needed.
//!
//! The bug in #4327 was caused by dvoice's `RopePermutingBackend` still
//! permuting Q/K weights for the OLD interleaved convention after nn-qwen3
//! switched to half-split in commit 62ddb7398.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::RotaryEmbedding;
use nn_core::Device;

/// Compute the HuggingFace `rotate_half` reference output for a single
/// position on a single head vector.
///
/// This is a direct translation of PyTorch's:
/// ```python
/// def rotate_half(x):
///     x1 = x[..., :half]
///     x2 = x[..., half:]
///     return torch.cat((-x2, x1), dim=-1)
///
/// def apply_rotary_pos_emb(q, cos, sin, position_ids):
///     cos = cos[position_ids]
///     sin = sin[position_ids]
///     q_embed = (q * cos) + (rotate_half(q) * sin)
///     return q_embed
/// ```
fn reference_half_split_rope(x: &[f32], head_dim: usize, position: usize, base: f64) -> Vec<f32> {
    let half = head_dim / 2;
    assert_eq!(x.len(), head_dim);

    // Compute cos/sin for this position (same formula as RotaryEmbedding)
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for i in 0..half {
        let theta = 1.0 / base.powf((2 * i) as f64 / head_dim as f64);
        let angle = position as f64 * theta;
        cos[i] = angle.cos() as f32;
        sin[i] = angle.sin() as f32;
    }

    // x1 = x[:half], x2 = x[half:]
    let x1 = &x[..half];
    let x2 = &x[half..];

    // y1 = x1 * cos - x2 * sin
    // y2 = x1 * sin + x2 * cos
    let mut result = vec![0.0f32; head_dim];
    for i in 0..half {
        result[i] = x1[i] * cos[i] - x2[i] * sin[i];
        result[i + half] = x1[i] * sin[i] + x2[i] * cos[i];
    }
    result
}

/// Permute a head vector from half-split to interleaved convention.
///
/// This is what `RopePermutingBackend` does to Q/K weight rows:
/// `new[2i] = old[i], new[2i+1] = old[i+half]`
fn permute_half_split_to_interleaved(x: &[f32], head_dim: usize) -> Vec<f32> {
    let half = head_dim / 2;
    let mut permuted = vec![0.0f32; head_dim];
    for i in 0..half {
        permuted[2 * i] = x[i];
        permuted[2 * i + 1] = x[i + half];
    }
    permuted
}

// ---------------------------------------------------------------------------
// Half-split RoPE matches HuggingFace rotate_half reference
// ---------------------------------------------------------------------------

#[test]
fn test_half_split_rope_matches_huggingface_reference_head8() {
    verify_half_split_matches_reference(8, 10_000.0, &[1, 5, 10, 42]);
}

#[test]
fn test_half_split_rope_matches_huggingface_reference_head128() {
    // Qwen3 uses head_dim=128, base=1_000_000
    verify_half_split_matches_reference(128, 1_000_000.0, &[0, 1, 7, 42, 63]);
}

#[test]
fn test_half_split_rope_matches_huggingface_reference_head128_base10k() {
    verify_half_split_matches_reference(128, 10_000.0, &[0, 1, 50, 100]);
}

fn verify_half_split_matches_reference(head_dim: usize, base: f64, positions: &[usize]) {
    let max_seq_len = positions.iter().copied().max().unwrap_or(0) + 1;
    let rope = RotaryEmbedding::new(head_dim, max_seq_len, base, &Device::Cpu).unwrap();

    // Deterministic input vector
    let input_data: Vec<f32> = (0..head_dim)
        .map(|i| ((i as f32 + 1.0) * 0.1).sin())
        .collect();

    for &pos in positions {
        // Compute reference output
        let expected = reference_half_split_rope(&input_data, head_dim, pos, base);

        // Compute nn output using apply_pair_half_split
        // Input shape: [1, 1, 1, head_dim] (batch=1, heads=1, seq=1, head_dim)
        let q =
            DynTensor::from_vec(input_data.clone(), &[1, 1, 1, head_dim], &Device::Cpu).unwrap();
        let k = q.clone();
        let (q_rot, _k_rot) = rope.apply_pair_half_split(&q, &k, &[pos]).unwrap();
        let q_flat = q_rot.to_flat_vec::<f32>().unwrap();

        assert_eq!(
            q_flat.len(),
            expected.len(),
            "output length mismatch at pos={pos}"
        );

        let max_diff: f32 = q_flat
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "half-split RoPE diverges from HF reference at pos={pos}: max_diff={max_diff} \
             (head_dim={head_dim}, base={base})"
        );
    }
}

// ---------------------------------------------------------------------------
// Regression: permuted weights + half-split RoPE produces WRONG results
// ---------------------------------------------------------------------------
//
// This is the exact bug pattern from #4327. If someone permutes Q/K weights
// from half-split to interleaved convention and then applies half-split RoPE,
// the rotation is mathematically wrong and produces large divergence.

#[test]
fn test_permuted_weights_with_half_split_rope_diverges_from_reference() {
    let head_dim = 128;
    let base = 1_000_000.0;
    let pos = 42;

    // Deterministic input
    let input_data: Vec<f32> = (0..head_dim)
        .map(|i| ((i as f32 + 1.0) * 0.1).sin())
        .collect();

    // Reference: half-split RoPE on unpermuted weights (correct)
    let correct = reference_half_split_rope(&input_data, head_dim, pos, base);

    // Bug pattern: permute weights (half-split -> interleaved) then apply half-split RoPE
    let permuted = permute_half_split_to_interleaved(&input_data, head_dim);
    let wrong = reference_half_split_rope(&permuted, head_dim, pos, base);

    // The permuted result should NOT match the correct result
    let max_diff: f32 = correct
        .iter()
        .zip(wrong.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff > 0.01,
        "permuted weights + half-split RoPE should diverge from correct output, \
         but max_diff={max_diff}. This means the regression test is not catching the #4327 bug."
    );
}

// ---------------------------------------------------------------------------
// Verify interleaved RoPE on permuted weights matches half-split on original
// ---------------------------------------------------------------------------
//
// Sanity check: the two conventions ARE equivalent when weights match.

#[test]
fn test_interleaved_on_permuted_equals_half_split_on_original() {
    let head_dim = 8;
    let base = 10_000.0;
    let half = head_dim / 2;

    for pos in [1, 5, 20] {
        let input_data: Vec<f32> = (0..head_dim)
            .map(|i| ((i as f32 + 1.0) * 0.3).cos())
            .collect();

        // Half-split RoPE on original data (correct for HF weights)
        let half_split_result = reference_half_split_rope(&input_data, head_dim, pos, base);

        // Interleaved RoPE on permuted data (correct for permuted weights)
        let permuted = permute_half_split_to_interleaved(&input_data, head_dim);
        let interleaved_result = reference_interleaved_rope(&permuted, head_dim, pos, base);

        // Un-permute the interleaved result back to half-split ordering
        // for comparison: inverse permutation is:
        // orig[i] = permuted[2i], orig[i+half] = permuted[2i+1]
        let mut unpermuted = vec![0.0f32; head_dim];
        for i in 0..half {
            unpermuted[i] = interleaved_result[2 * i];
            unpermuted[i + half] = interleaved_result[2 * i + 1];
        }

        let max_diff: f32 = half_split_result
            .iter()
            .zip(unpermuted.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "interleaved on permuted should match half-split on original \
             at pos={pos}, but max_diff={max_diff}"
        );
    }
}

/// Reference interleaved RoPE: pairs (2i, 2i+1).
fn reference_interleaved_rope(x: &[f32], head_dim: usize, position: usize, base: f64) -> Vec<f32> {
    let half = head_dim / 2;
    assert_eq!(x.len(), head_dim);

    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for i in 0..half {
        let theta = 1.0 / base.powf((2 * i) as f64 / head_dim as f64);
        let angle = position as f64 * theta;
        cos[i] = angle.cos() as f32;
        sin[i] = angle.sin() as f32;
    }

    let mut result = vec![0.0f32; head_dim];
    for i in 0..half {
        result[2 * i] = x[2 * i] * cos[i] - x[2 * i + 1] * sin[i];
        result[2 * i + 1] = x[2 * i] * sin[i] + x[2 * i + 1] * cos[i];
    }
    result
}
