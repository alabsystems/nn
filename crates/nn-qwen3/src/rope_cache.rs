// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Precomputed RoPE (Rotary Position Embedding) cache for Qwen3.
//!
//! [`RoPECache`] precomputes sin/cos frequency tables for all positions
//! up to `max_seq_len`, enabling O(1) lookup per position during inference
//! instead of recomputing trigonometric functions on every forward pass.
//!
//! This operates on raw `f32` slices — complementary to the [`DynTensor`]-based
//! [`RotaryEmbedding`](nn_core::layers::RotaryEmbedding) in nn-core. Use this
//! when you need direct `f32` access (e.g., custom kernels, verification,
//! manual SIMD paths).
//!
//! # Example
//!
//! ```
//! use nn_qwen3::rope_cache::RoPECache;
//!
//! let cache = RoPECache::new(2048, 128, 10_000.0);
//! let (cos, sin) = cache.get(42);
//! assert_eq!(cos.len(), 64); // head_dim / 2
//!
//! // Apply RoPE in-place to query and key vectors
//! let mut q = vec![1.0f32; 128];
//! let mut k = vec![0.5f32; 128];
//! RoPECache::apply_rope(&mut q, &mut k, cos, sin);
//! ```

use crate::Qwen3Error;
use nn_core::Result;

/// Precomputed sin/cos cache for Rotary Position Embeddings.
///
/// Stores `cos(p * theta[i])` and `sin(p * theta[i])` for every position
/// `p` in `0..max_seq_len` and every frequency index `i` in `0..head_dim/2`,
/// where `theta[i] = 1.0 / base^(2i / head_dim)`.
///
/// # Memory
///
/// Uses `2 * max_seq_len * (head_dim / 2) * 4` bytes. For Qwen3 defaults
/// (max_seq_len=32768, head_dim=128): ~8 MiB.
#[derive(Debug, Clone)]
pub struct RoPECache {
    /// `cos_cache[p]` has length `head_dim / 2`.
    cos_cache: Vec<Vec<f32>>,
    /// `sin_cache[p]` has length `head_dim / 2`.
    sin_cache: Vec<Vec<f32>>,
    /// Half of head_dim (number of frequency pairs).
    half_dim: usize,
    /// Maximum sequence length this cache supports.
    max_seq_len: usize,
    /// Head dimension this cache was built for.
    head_dim: usize,
    /// Base frequency used to compute theta values.
    base: f32,
}

impl RoPECache {
    /// Create a new RoPE cache with precomputed sin/cos tables.
    ///
    /// - `max_seq_len`: number of positions to precompute (0..max_seq_len)
    /// - `head_dim`: attention head dimension (must be even and > 0)
    /// - `base`: RoPE base frequency (typically 10000.0 for LLaMA, 1000000.0 for Qwen3)
    ///
    /// # Panics
    ///
    /// Panics if `head_dim` is zero, odd, or `max_seq_len` is zero.
    #[must_use]
    pub fn new(max_seq_len: usize, head_dim: usize, base: f32) -> Self {
        assert!(
            head_dim > 0 && head_dim.is_multiple_of(2),
            "head_dim must be a positive even number"
        );
        assert!(max_seq_len > 0, "max_seq_len must be > 0");
        assert!(
            base > 0.0 && base.is_finite(),
            "base must be a positive finite number"
        );

        let half_dim = head_dim / 2;

        // theta[i] = 1.0 / base^(2i / head_dim)
        let theta: Vec<f64> = (0..half_dim)
            .map(|i| {
                let exponent = (2 * i) as f64 / head_dim as f64;
                1.0 / f64::from(base).powf(exponent)
            })
            .collect();

        let mut cos_cache = Vec::with_capacity(max_seq_len);
        let mut sin_cache = Vec::with_capacity(max_seq_len);

        for p in 0..max_seq_len {
            let mut cos_row = Vec::with_capacity(half_dim);
            let mut sin_row = Vec::with_capacity(half_dim);
            for &t in &theta {
                let angle = p as f64 * t;
                cos_row.push(angle.cos() as f32);
                sin_row.push(angle.sin() as f32);
            }
            cos_cache.push(cos_row);
            sin_cache.push(sin_row);
        }

        Self {
            cos_cache,
            sin_cache,
            half_dim,
            max_seq_len,
            head_dim,
            base,
        }
    }

    /// Get (cos, sin) slices for a single position.
    ///
    /// Each returned slice has length `head_dim / 2`.
    ///
    /// # Panics
    ///
    /// Panics if `position >= max_seq_len`.
    #[must_use]
    pub fn get(&self, position: usize) -> (&[f32], &[f32]) {
        assert!(
            position < self.max_seq_len,
            "position {position} >= max_seq_len {}",
            self.max_seq_len
        );
        (&self.cos_cache[position], &self.sin_cache[position])
    }

    /// Get (cos, sin) slices for a contiguous range of positions.
    ///
    /// Returns references to the internal `Vec<Vec<f32>>` slices for positions
    /// `start..start+len`.
    ///
    /// # Panics
    ///
    /// Panics if `start + len > max_seq_len`.
    #[must_use]
    pub fn get_range(&self, start: usize, len: usize) -> (&[Vec<f32>], &[Vec<f32>]) {
        let end = start + len;
        assert!(
            end <= self.max_seq_len,
            "start ({start}) + len ({len}) = {end} > max_seq_len {}",
            self.max_seq_len
        );
        (&self.cos_cache[start..end], &self.sin_cache[start..end])
    }

    /// Apply RoPE rotation in-place to query and key vectors.
    ///
    /// Both `q` and `k` must have length equal to `head_dim` (i.e., `2 * cos.len()`).
    /// `cos` and `sin` are the cached values for the target position, each of
    /// length `head_dim / 2`.
    ///
    /// The rotation operates on consecutive pairs `(x[2i], x[2i+1])`:
    ///
    /// ```text
    /// x_out[2i]   = x[2i]   * cos[i] - x[2i+1] * sin[i]
    /// x_out[2i+1] = x[2i]   * sin[i] + x[2i+1] * cos[i]
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `q.len() != 2 * cos.len()`, `k.len() != 2 * cos.len()`,
    /// or `cos.len() != sin.len()`.
    pub fn apply_rope(q: &mut [f32], k: &mut [f32], cos: &[f32], sin: &[f32]) {
        let half_dim = cos.len();
        assert_eq!(sin.len(), half_dim, "cos and sin must have equal length");
        assert_eq!(
            q.len(),
            2 * half_dim,
            "q length ({}) must equal 2 * half_dim ({})",
            q.len(),
            2 * half_dim
        );
        assert_eq!(
            k.len(),
            2 * half_dim,
            "k length ({}) must equal 2 * half_dim ({})",
            k.len(),
            2 * half_dim
        );

        Self::rotate_vector(q, cos, sin);
        Self::rotate_vector(k, cos, sin);
    }

    /// Rotate a single vector in-place using precomputed cos/sin.
    fn rotate_vector(x: &mut [f32], cos: &[f32], sin: &[f32]) {
        let half_dim = cos.len();
        for i in 0..half_dim {
            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];
            x[2 * i] = x0 * cos[i] - x1 * sin[i];
            x[2 * i + 1] = x0 * sin[i] + x1 * cos[i];
        }
    }

    /// Validated constructor returning `Result` instead of panicking.
    ///
    /// Prefer this in contexts where invalid parameters should produce
    /// error values rather than panics (e.g., model loading from untrusted config).
    pub fn try_new(max_seq_len: usize, head_dim: usize, base: f32) -> Result<Self> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(Qwen3Error::InvalidConfig {
                reason: format!("head_dim must be a positive even number, got {head_dim}"),
            }
            .into());
        }
        if max_seq_len == 0 {
            return Err(Qwen3Error::InvalidConfig {
                reason: "max_seq_len must be > 0".into(),
            }
            .into());
        }
        if !base.is_finite() || base <= 0.0 {
            return Err(Qwen3Error::InvalidConfig {
                reason: format!("base must be a positive finite number, got {base}"),
            }
            .into());
        }
        Ok(Self::new(max_seq_len, head_dim, base))
    }

    /// Half of head_dim (number of frequency pairs per position).
    #[must_use]
    pub fn half_dim(&self) -> usize {
        self.half_dim
    }

    /// Head dimension this cache was built for.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Maximum sequence length this cache supports.
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Base frequency used to compute theta values.
    #[must_use]
    pub fn base(&self) -> f32 {
        self.base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: compute cos/sin for a single position and frequency index
    /// from scratch (no cache), for comparison.
    fn reference_cos_sin(
        position: usize,
        freq_idx: usize,
        head_dim: usize,
        base: f32,
    ) -> (f32, f32) {
        let exponent = (2 * freq_idx) as f64 / head_dim as f64;
        let theta = 1.0 / f64::from(base).powf(exponent);
        let angle = position as f64 * theta;
        (angle.cos() as f32, angle.sin() as f32)
    }

    // -- Cache produces correct sin/cos for known positions --

    #[test]
    fn test_position_zero_is_identity() {
        // At position 0, all angles are 0 => cos=1, sin=0.
        let cache = RoPECache::new(128, 128, 10_000.0);
        let (cos, sin) = cache.get(0);
        for i in 0..64 {
            assert!(
                (cos[i] - 1.0).abs() < 1e-7,
                "cos[{i}] at pos 0 should be 1.0, got {}",
                cos[i]
            );
            assert!(
                sin[i].abs() < 1e-7,
                "sin[{i}] at pos 0 should be 0.0, got {}",
                sin[i]
            );
        }
    }

    #[test]
    fn test_known_position_values() {
        let cache = RoPECache::new(256, 64, 10_000.0);
        // Spot-check several (position, frequency_index) pairs.
        for pos in [1, 7, 42, 100, 255] {
            for freq_idx in [0, 5, 15, 31] {
                let (ref_cos, ref_sin) = reference_cos_sin(pos, freq_idx, 64, 10_000.0);
                let (cos, sin) = cache.get(pos);
                assert!(
                    (cos[freq_idx] - ref_cos).abs() < 1e-5,
                    "cos mismatch at pos={pos}, idx={freq_idx}: cache={}, ref={ref_cos}",
                    cos[freq_idx]
                );
                assert!(
                    (sin[freq_idx] - ref_sin).abs() < 1e-5,
                    "sin mismatch at pos={pos}, idx={freq_idx}: cache={}, ref={ref_sin}",
                    sin[freq_idx]
                );
            }
        }
    }

    // -- apply_rope correctly rotates vectors --

    #[test]
    fn test_apply_rope_identity_at_position_zero() {
        // At position 0: cos=1, sin=0, so rotation is identity.
        let cache = RoPECache::new(16, 8, 10_000.0);
        let (cos, sin) = cache.get(0);

        let original_q: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let original_k: Vec<f32> = (0..8).map(|i| (i + 10) as f32).collect();
        let mut q = original_q.clone();
        let mut k = original_k.clone();

        RoPECache::apply_rope(&mut q, &mut k, cos, sin);

        for i in 0..8 {
            assert!(
                (q[i] - original_q[i]).abs() < 1e-5,
                "q[{i}]: expected {}, got {}",
                original_q[i],
                q[i]
            );
            assert!(
                (k[i] - original_k[i]).abs() < 1e-5,
                "k[{i}]: expected {}, got {}",
                original_k[i],
                k[i]
            );
        }
    }

    #[test]
    fn test_apply_rope_rotation_correctness() {
        // Manually verify rotation formula for a small case.
        let head_dim = 4;
        let cache = RoPECache::new(16, head_dim, 10_000.0);
        let (cos, sin) = cache.get(5);

        let mut q = [3.0f32, 4.0, 1.0, 2.0];
        let mut k = [5.0f32, 6.0, 7.0, 8.0];

        RoPECache::apply_rope(&mut q, &mut k, cos, sin);

        // For pair (q[0], q[1]) with (cos[0], sin[0]):
        //   q_out[0] = 3.0 * cos[0] - 4.0 * sin[0]
        //   q_out[1] = 3.0 * sin[0] + 4.0 * cos[0]
        let expected_q0 = 3.0 * cos[0] - 4.0 * sin[0];
        let expected_q1 = 3.0 * sin[0] + 4.0 * cos[0];
        let expected_q2 = 1.0 * cos[1] - 2.0 * sin[1];
        let expected_q3 = 1.0 * sin[1] + 2.0 * cos[1];

        assert!((q[0] - expected_q0).abs() < 1e-6, "q[0] mismatch");
        assert!((q[1] - expected_q1).abs() < 1e-6, "q[1] mismatch");
        assert!((q[2] - expected_q2).abs() < 1e-6, "q[2] mismatch");
        assert!((q[3] - expected_q3).abs() < 1e-6, "q[3] mismatch");

        // Same check for k.
        let expected_k0 = 5.0 * cos[0] - 6.0 * sin[0];
        let expected_k1 = 5.0 * sin[0] + 6.0 * cos[0];
        assert!((k[0] - expected_k0).abs() < 1e-6, "k[0] mismatch");
        assert!((k[1] - expected_k1).abs() < 1e-6, "k[1] mismatch");
    }

    #[test]
    fn test_apply_rope_preserves_norm() {
        // RoPE is an orthogonal rotation, so it preserves vector norm.
        let cache = RoPECache::new(128, 64, 10_000.0);
        let (cos, sin) = cache.get(42);

        let mut q: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut k: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) * 0.2).collect();

        let q_norm_before: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let k_norm_before: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();

        RoPECache::apply_rope(&mut q, &mut k, cos, sin);

        let q_norm_after: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let k_norm_after: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (q_norm_before - q_norm_after).abs() < 1e-4,
            "q norm changed: {q_norm_before} -> {q_norm_after}"
        );
        assert!(
            (k_norm_before - k_norm_after).abs() < 1e-4,
            "k norm changed: {k_norm_before} -> {k_norm_after}"
        );
    }

    // -- Cached values match fresh computation --

    #[test]
    fn test_cached_matches_fresh_computation_head64() {
        verify_cache_matches_fresh(64, 10_000.0, 512);
    }

    #[test]
    fn test_cached_matches_fresh_computation_head128() {
        verify_cache_matches_fresh(128, 10_000.0, 1024);
    }

    #[test]
    fn test_cached_matches_fresh_computation_head256() {
        verify_cache_matches_fresh(256, 10_000.0, 256);
    }

    #[test]
    fn test_cached_matches_fresh_qwen3_base() {
        // Qwen3 uses base=1_000_000
        verify_cache_matches_fresh(128, 1_000_000.0, 512);
    }

    fn verify_cache_matches_fresh(head_dim: usize, base: f32, max_seq_len: usize) {
        let cache = RoPECache::new(max_seq_len, head_dim, base);
        let half_dim = head_dim / 2;

        // Check every 50th position (to keep test fast).
        let positions: Vec<usize> = (0..max_seq_len).step_by(50).collect();
        for &pos in &positions {
            let (cos, sin) = cache.get(pos);
            assert_eq!(cos.len(), half_dim);
            assert_eq!(sin.len(), half_dim);
            for freq_idx in 0..half_dim {
                let (ref_cos, ref_sin) = reference_cos_sin(pos, freq_idx, head_dim, base);
                assert!(
                    (cos[freq_idx] - ref_cos).abs() < 1e-5,
                    "cos mismatch: head_dim={head_dim}, base={base}, pos={pos}, idx={freq_idx}: got {}, expected {ref_cos}",
                    cos[freq_idx]
                );
                assert!(
                    (sin[freq_idx] - ref_sin).abs() < 1e-5,
                    "sin mismatch: head_dim={head_dim}, base={base}, pos={pos}, idx={freq_idx}: got {}, expected {ref_sin}",
                    sin[freq_idx]
                );
            }
        }
    }

    // -- Various head_dim sizes: 64, 128, 256 --

    #[test]
    fn test_head_dim_64() {
        let cache = RoPECache::new(64, 64, 10_000.0);
        assert_eq!(cache.head_dim(), 64);
        assert_eq!(cache.half_dim(), 32);
        let (cos, sin) = cache.get(0);
        assert_eq!(cos.len(), 32);
        assert_eq!(sin.len(), 32);
    }

    #[test]
    fn test_head_dim_128() {
        let cache = RoPECache::new(64, 128, 10_000.0);
        assert_eq!(cache.head_dim(), 128);
        assert_eq!(cache.half_dim(), 64);
        let (cos, sin) = cache.get(63);
        assert_eq!(cos.len(), 64);
        assert_eq!(sin.len(), 64);
    }

    #[test]
    fn test_head_dim_256() {
        let cache = RoPECache::new(32, 256, 10_000.0);
        assert_eq!(cache.head_dim(), 256);
        assert_eq!(cache.half_dim(), 128);
        let (cos, sin) = cache.get(31);
        assert_eq!(cos.len(), 128);
        assert_eq!(sin.len(), 128);
    }

    // -- get_range tests --

    #[test]
    fn test_get_range_matches_individual_gets() {
        let cache = RoPECache::new(100, 64, 10_000.0);
        let (cos_range, sin_range) = cache.get_range(10, 5);
        assert_eq!(cos_range.len(), 5);
        assert_eq!(sin_range.len(), 5);
        for i in 0..5 {
            let (cos_single, sin_single) = cache.get(10 + i);
            assert_eq!(cos_range[i].as_slice(), cos_single);
            assert_eq!(sin_range[i].as_slice(), sin_single);
        }
    }

    #[test]
    fn test_get_range_full() {
        let cache = RoPECache::new(16, 8, 10_000.0);
        let (cos_range, sin_range) = cache.get_range(0, 16);
        assert_eq!(cos_range.len(), 16);
        assert_eq!(sin_range.len(), 16);
    }

    // -- try_new error cases --

    #[test]
    fn test_try_new_rejects_zero_head_dim() {
        let result = RoPECache::try_new(128, 0, 10_000.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_new_rejects_odd_head_dim() {
        let result = RoPECache::try_new(128, 7, 10_000.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_new_rejects_zero_max_seq_len() {
        let result = RoPECache::try_new(0, 128, 10_000.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_new_rejects_negative_base() {
        let result = RoPECache::try_new(128, 128, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_new_rejects_nan_base() {
        let result = RoPECache::try_new(128, 128, f32::NAN);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_new_rejects_inf_base() {
        let result = RoPECache::try_new(128, 128, f32::INFINITY);
        assert!(result.is_err());
    }

    // -- accessor tests --

    #[test]
    fn test_accessors() {
        let cache = RoPECache::new(2048, 128, 1_000_000.0);
        assert_eq!(cache.max_seq_len(), 2048);
        assert_eq!(cache.head_dim(), 128);
        assert_eq!(cache.half_dim(), 64);
        assert!((cache.base() - 1_000_000.0).abs() < f32::EPSILON);
    }

    // -- panic tests --

    #[test]
    #[should_panic(expected = "position 16 >= max_seq_len 16")]
    fn test_get_panics_on_out_of_bounds() {
        let cache = RoPECache::new(16, 8, 10_000.0);
        let _ = cache.get(16);
    }

    #[test]
    #[should_panic(expected = "head_dim must be a positive even number")]
    fn test_new_panics_on_odd_head_dim() {
        let _ = RoPECache::new(16, 3, 10_000.0);
    }

    #[test]
    #[should_panic(expected = "q length (4) must equal 2 * half_dim (8)")]
    fn test_apply_rope_panics_on_mismatched_q_len() {
        let cos = [1.0f32; 4];
        let sin = [0.0f32; 4];
        let mut q = [0.0f32; 4]; // wrong: should be 8
        let mut k = [0.0f32; 8];
        RoPECache::apply_rope(&mut q, &mut k, &cos, &sin);
    }
}
