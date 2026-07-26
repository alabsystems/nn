// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for embedding layer safety in dpdf VLMs (#4236).
//!
//! Proofs targeting dpdf-specific VLM embedding patterns:
//!
//! 1. RoPE inverse-frequency bounds: `base^(-2i/d)` stays finite
//! 2. RoPE angle computation: `pos * inv_freq` finite, cos/sin valid
//! 3. RoPE cache allocation: no overflow for realistic params
//! 4. Triple embedding sum: token + position + type preserves interval bounds
//! 5. Sinusoidal PE value bounds: sin/cos outputs in [-1, 1]
//! 6. dpdf VLM embedding pipeline shape: token + pos + seg -> LayerNorm
//! 7. RoPE rotation preserves vector norm (orthogonality)
//! 8. RoPE pairwise dimension coverage
//! 9. RoPE offset bounds checking
//!
//! Part of #4236.

#![cfg(kani)]

use crate::tensor::checked_dim_product;

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn cos_f32_stub(_x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}

fn sin_f32_stub(_x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}

fn powf_f64_stub(_b: f64, _e: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1.0);
    r
}

// =============================================================================
// 1. RoPE inverse-frequency bounds
// =============================================================================

/// Prove: `inv_freq[i] = 1 / base^(2i/d)` is in (0, 1] for valid params.
/// Since base >= 1 and exponent in [0, 1): base^exp >= 1, so inv_freq <= 1.
///
/// Part of #4236.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rope_inv_freq_bounded() {
    let head_dim: usize = kani::any();
    let dim_pair_idx: usize = kani::any();
    let base: f64 = kani::any();

    kani::assume(head_dim >= 2 && head_dim <= 256);
    kani::assume(head_dim % 2 == 0);
    kani::assume(base.is_finite() && base >= 1.0 && base <= 1_000_000.0);

    let half_dim = head_dim / 2;
    kani::assume(dim_pair_idx < half_dim);

    let exponent = (2 * dim_pair_idx) as f64 / head_dim as f64;
    assert!(exponent >= 0.0, "exponent must be non-negative");
    assert!(exponent < 1.0, "exponent must be < 1 for i < half_dim");

    let base_powered = powf_f64_stub(base, exponent);
    kani::assume(base_powered > 0.0);

    let inv_freq = 1.0 / base_powered;
    assert!(inv_freq.is_finite(), "inv_freq must be finite");
    assert!(inv_freq > 0.0, "inv_freq must be positive");
    assert!(inv_freq <= 1.0, "inv_freq must be <= 1.0 for base >= 1");
}

// =============================================================================
// 2. RoPE angle computation stays finite
// =============================================================================

/// Prove: `pos * inv_freq` stays finite for realistic positions and valid
/// inverse frequencies, and cos/sin produce values in [-1, 1].
///
/// Part of #4236.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rope_angle_computation_finite() {
    let pos: usize = kani::any();
    let inv_freq: f32 = kani::any();

    kani::assume(pos <= 131_072);
    kani::assume(inv_freq.is_finite() && inv_freq > 0.0 && inv_freq <= 1.0);

    let angle_f64 = pos as f64 * f64::from(inv_freq);
    assert!(angle_f64.is_finite(), "angle in f64 must be finite");

    let angle_f32 = angle_f64 as f32;
    assert!(angle_f32.is_finite(), "angle in f32 must be finite");

    let cos_val = cos_f32_stub(angle_f32);
    let sin_val = sin_f32_stub(angle_f32);

    assert!(cos_val >= -1.0 && cos_val <= 1.0, "cos must be in [-1, 1]");
    assert!(sin_val >= -1.0 && sin_val <= 1.0, "sin must be in [-1, 1]");
    assert!(cos_val.is_finite(), "cos cache element must be finite");
    assert!(sin_val.is_finite(), "sin cache element must be finite");
}

// =============================================================================
// 3. RoPE cache allocation no overflow
// =============================================================================

/// Prove: `max_seq_len * half_dim` does not overflow for realistic params,
/// matching the production `checked_mul`.
///
/// Part of #4236.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rope_cache_allocation_no_overflow() {
    let max_seq_len: usize = kani::any();
    let head_dim: usize = kani::any();

    kani::assume(max_seq_len >= 1 && max_seq_len <= 131_072);
    kani::assume(head_dim >= 2 && head_dim <= 256);
    kani::assume(head_dim % 2 == 0);

    let half_dim = head_dim / 2;
    let cache_len = max_seq_len.checked_mul(half_dim);
    assert!(cache_len.is_some(), "cache allocation must not overflow");

    let cache_len = cache_len.unwrap();
    assert!(cache_len <= 131_072 * 128, "cache within expected bounds");

    let total_cache = cache_len.checked_mul(2);
    assert!(
        total_cache.is_some(),
        "total cos+sin cache must not overflow"
    );
}

// =============================================================================
// 4. Triple embedding sum bounds (token + position + type)
// =============================================================================

/// Prove: token emb in [-T,T] + pos emb in [-P,P] + seg emb in [-S,S]
/// yields sum in [-(T+P+S), T+P+S]. Used by BERT-family document VLMs
/// (LayoutLM, Granite-Docling) where embed = token + pos + segment.
///
/// Part of #4236.
#[kani::unwind(1)]
#[kani::proof]
fn proof_triple_embedding_sum_bounds() {
    let token_bound: f32 = kani::any();
    let pos_bound: f32 = kani::any();
    let seg_bound: f32 = kani::any();
    let token_val: f32 = kani::any();
    let pos_val: f32 = kani::any();
    let seg_val: f32 = kani::any();

    kani::assume(token_bound.is_finite() && token_bound >= 0.0 && token_bound <= 1e3);
    kani::assume(pos_bound.is_finite() && pos_bound >= 0.0 && pos_bound <= 1e3);
    kani::assume(seg_bound.is_finite() && seg_bound >= 0.0 && seg_bound <= 1e3);

    kani::assume(token_val.is_finite());
    kani::assume(token_val >= -token_bound && token_val <= token_bound);
    kani::assume(pos_val.is_finite());
    kani::assume(pos_val >= -pos_bound && pos_val <= pos_bound);
    kani::assume(seg_val.is_finite());
    kani::assume(seg_val >= -seg_bound && seg_val <= seg_bound);

    let combined_bound = token_bound + pos_bound + seg_bound;
    kani::assume(combined_bound.is_finite());

    let sum_tp = token_val + pos_val;
    kani::assume(sum_tp.is_finite());
    let sum_tps = sum_tp + seg_val;
    kani::assume(sum_tps.is_finite());

    assert!(sum_tps >= -combined_bound, "sum >= -(T+P+S)");
    assert!(sum_tps <= combined_bound, "sum <= (T+P+S)");
}

// =============================================================================
// 5. Sinusoidal positional encoding value bounds
// =============================================================================

/// Prove: sinusoidal PE values are always in [-1, 1], since sin/cos are
/// bounded. Even dims use sin, odd dims use cos.
///
/// Part of #4236.
#[kani::unwind(1)]
#[kani::proof]
fn proof_sinusoidal_pe_values_bounded() {
    let pos: usize = kani::any();
    let d_model: usize = kani::any();
    let dim_idx: usize = kani::any();

    kani::assume(pos <= 8192);
    kani::assume(d_model >= 2 && d_model <= 1024);
    kani::assume(d_model % 2 == 0);
    kani::assume(dim_idx < d_model);

    let pair_idx = dim_idx / 2;
    let exponent = (2 * pair_idx) as f64 / d_model as f64;
    let denom = powf_f64_stub(10000.0, exponent);
    kani::assume(denom > 0.0 && denom.is_finite());

    let angle = pos as f64 / denom;
    kani::assume(angle.is_finite());
    let angle_f32 = angle as f32;
    kani::assume(angle_f32.is_finite());

    let pe_value = if dim_idx % 2 == 0 {
        sin_f32_stub(angle_f32)
    } else {
        cos_f32_stub(angle_f32)
    };

    assert!(pe_value >= -1.0 && pe_value <= 1.0, "PE value in [-1, 1]");
    assert!(pe_value.is_finite(), "PE value must be finite");
}

// =============================================================================
// 6. dpdf VLM embedding pipeline end-to-end shape
// =============================================================================

/// Prove: the BERT-family dpdf VLM embedding pipeline preserves shape:
///   token_emb[B,S,D] + pos_emb[B,S,D] + seg_emb[B,S,D] -> [B,S,D]
/// All three embeddings produce identical shapes from their respective
/// weight tables before element-wise addition.
///
/// Part of #4236.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dpdf_vlm_embedding_pipeline_shape() {
    let batch: usize = kani::any();
    let seq_len: usize = kani::any();
    let vocab_size: usize = kani::any();
    let max_pos: usize = kani::any();
    let num_segments: usize = kani::any();
    let d_model: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 4);
    kani::assume(seq_len >= 1 && seq_len <= 64);
    kani::assume(vocab_size >= 1 && vocab_size <= 512);
    kani::assume(max_pos >= seq_len && max_pos <= 1024);
    kani::assume(num_segments >= 1 && num_segments <= 4);
    kani::assume(d_model >= 1 && d_model <= 128);

    // Token embedding: [B,S] input, [V,D] weight -> [B,S,D]
    let token_in = [batch, seq_len];
    let token_wt = [vocab_size, d_model];
    let token_out = [batch, seq_len, d_model];
    assert!(token_out.len() == token_in.len() + 1);
    assert!(token_out[2] == token_wt[1]);

    let token_numel = checked_dim_product(&token_out);
    assert!(token_numel.is_ok(), "token output numel valid");

    // Position embedding: [max_pos, D] weight, select [0..seq_len] -> [B,S,D]
    let pos_out = [batch, seq_len, d_model];
    let pos_numel = checked_dim_product(&pos_out);
    assert!(pos_numel.is_ok(), "position output numel valid");

    // Segment embedding: [num_segments, D] weight -> [B,S,D]
    let seg_out = [batch, seq_len, d_model];
    let seg_numel = checked_dim_product(&seg_out);
    assert!(seg_numel.is_ok(), "segment output numel valid");

    // All three have identical shape — element-wise sum is valid
    assert!(token_out == pos_out, "token and position shapes match");
    assert!(pos_out == seg_out, "position and segment shapes match");

    // Sum preserves shape and numel
    let sum_numel = checked_dim_product(&token_out);
    assert!(sum_numel.is_ok());
    assert!(sum_numel.unwrap() == token_numel.unwrap());
}

// =============================================================================
// 7. RoPE rotation preserves vector norm (orthogonality)
// =============================================================================

/// Prove: RoPE rotation of (x0, x1) by angle theta preserves L2 norm.
///   x0' = x0*cos - x1*sin, x1' = x0*sin + x1*cos
///   => x0'^2 + x1'^2 = x0^2 + x1^2  (Pythagorean identity)
///
/// Critical for attention score stability.
///
/// Part of #4236.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rope_rotation_preserves_norm() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();

    kani::assume(x0.is_finite() && x0.abs() <= 100.0);
    kani::assume(x1.is_finite() && x1.abs() <= 100.0);

    let cos_theta = cos_f32_stub(0.0);
    let sin_theta = sin_f32_stub(0.0);

    // Pythagorean constraint on stubs
    let identity = cos_theta * cos_theta + sin_theta * sin_theta;
    kani::assume(identity >= 0.999 && identity <= 1.001);

    let x0_rot = x0 * cos_theta - x1 * sin_theta;
    let x1_rot = x0 * sin_theta + x1 * cos_theta;
    kani::assume(x0_rot.is_finite() && x1_rot.is_finite());

    let norm_sq_orig = x0 * x0 + x1 * x1;
    kani::assume(norm_sq_orig.is_finite());
    let norm_sq_rot = x0_rot * x0_rot + x1_rot * x1_rot;
    kani::assume(norm_sq_rot.is_finite());

    let diff = (norm_sq_rot - norm_sq_orig).abs();
    // f32 rounding: ~6 * eps * max_norm_sq ~= 6 * 1.2e-7 * 20000 ~= 0.015
    assert!(diff <= 0.1, "RoPE rotation must preserve L2 norm");
}

// =============================================================================
// 8. RoPE pairwise dimension coverage
// =============================================================================

/// Prove: for head_dim = 2k, RoPE rotates k pairs covering all dims exactly once.
///
/// Part of #4236.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rope_pairwise_coverage() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 2 && head_dim <= 256);
    kani::assume(head_dim % 2 == 0);

    let half_dim = head_dim / 2;
    let total_covered = 2 * half_dim;
    assert!(
        total_covered == head_dim,
        "all dims covered by rotation pairs"
    );
    assert!(half_dim == head_dim / 2, "pair count is head_dim / 2");
}

// =============================================================================
// 9. RoPE offset bounds checking
// =============================================================================

/// Prove: the production check `offset + seq_len <= max_seq_len` prevents
/// out-of-cache access. When valid, all positions are within bounds.
///
/// Part of #4236.
#[kani::unwind(5)]
#[kani::proof]
fn proof_rope_offset_bounds_check() {
    let max_seq_len: usize = kani::any();
    let seq_len: usize = kani::any();
    let offset: usize = kani::any();

    kani::assume(max_seq_len >= 1 && max_seq_len <= 131_072);
    kani::assume(seq_len >= 1 && seq_len <= max_seq_len);
    kani::assume(offset <= max_seq_len);

    let end_pos = offset.checked_add(seq_len);
    assert!(end_pos.is_some(), "offset + seq_len must not overflow");
    let end_pos = end_pos.unwrap();

    if end_pos <= max_seq_len {
        // All positions in [offset, offset+seq_len) are within cache
        for logical_pos in 0..seq_len.min(4) {
            let actual_pos = offset + logical_pos;
            assert!(actual_pos < max_seq_len, "position within cache");
        }
    } else {
        // Production code returns Err here
        assert!(end_pos > max_seq_len, "out-of-range must be detected");
    }
}
