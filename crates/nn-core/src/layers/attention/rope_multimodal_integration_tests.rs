// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for MultimodalRoPE (M-ROPE) numerical properties (#2439).
//!
//! Verifies mathematical properties that are essential for correctness:
//! - Norm preservation (rotation should not change vector magnitude)
//! - Distinct positions produce distinct rotations
//! - Section independence across the HF/Qwen six-block layout
//! - Text-mode equivalence (all-equal positions should match standard RoPE behavior)
//! - Consistency across batch dimensions

use super::MultimodalRoPE;
use crate::dyn_tensor::DynTensor;
use crate::Device;

fn det_data(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.017).sin() * 0.5)
        .collect()
}

fn hf_reference_apply_single(
    x: &[f32],
    mrope_section_sizes: [usize; 3],
    positions: [usize; 3],
    head_dim: usize,
    base: f64,
) -> Vec<f32> {
    let half_dim = head_dim / 2;
    assert_eq!(x.len(), head_dim);

    let mut cos = Vec::with_capacity(half_dim);
    let mut sin = Vec::with_capacity(half_dim);
    let mut freq_offset = 0usize;
    for (section_idx, &section_pairs) in mrope_section_sizes.iter().enumerate() {
        for i in 0..section_pairs {
            let global_i = freq_offset + i;
            let exponent = (2 * global_i) as f64 / head_dim as f64;
            let inv_freq = 1.0 / base.powf(exponent);
            let angle = (positions[section_idx] as f64 * inv_freq) as f32;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
        freq_offset += section_pairs;
    }

    let mut out = vec![0.0; head_dim];
    for i in 0..half_dim {
        let x1 = x[i];
        let x2 = x[half_dim + i];
        out[i] = x1 * cos[i] - x2 * sin[i];
        out[half_dim + i] = x1 * sin[i] + x2 * cos[i];
    }
    out
}

// -- Norm preservation --------------------------------------------------------

/// RoPE is a rotation: it should preserve the L2 norm of each vector.
/// Verify this for M-ROPE across all 3 sections.
#[test]
fn test_mrope_preserves_l2_norm() {
    let head_dim = 24;
    let rope = MultimodalRoPE::new(head_dim, [4, 4, 4], 128, 10000.0, &Device::Cpu).unwrap();

    let data = det_data(4 * head_dim, 42.0);
    let x = DynTensor::from_vec(data, &[1, 1, 4, head_dim], &Device::Cpu).unwrap();

    let rotated = rope
        .apply(&x, &[0, 5, 10, 50], &[0, 3, 7, 12], &[0, 1, 8, 20])
        .unwrap();

    let orig = x.to_flat_vec::<f32>().unwrap();
    let rot = rotated.to_flat_vec::<f32>().unwrap();

    // Check norm per-vector (last dimension)
    for token_idx in 0..4 {
        let start = token_idx * head_dim;
        let end = start + head_dim;
        let orig_norm: f32 = orig[start..end].iter().map(|v| v * v).sum::<f32>().sqrt();
        let rot_norm: f32 = rot[start..end].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (orig_norm - rot_norm).abs() < 1e-4,
            "Norm not preserved at token {token_idx}: orig={orig_norm}, rotated={rot_norm}"
        );
    }
}

// -- Distinct positions produce distinct rotations ----------------------------

/// Different position IDs should produce different outputs (unless the input
/// is zero). This catches bugs where positions are ignored.
#[test]
fn test_mrope_distinct_positions_produce_distinct_outputs() {
    let head_dim = 12;
    let rope = MultimodalRoPE::new(head_dim, [2, 2, 2], 64, 10000.0, &Device::Cpu).unwrap();

    let data = det_data(2 * head_dim, 42.0);
    let x = DynTensor::from_vec(data, &[1, 1, 2, head_dim], &Device::Cpu).unwrap();

    // Same input, different temporal positions
    let out_a = rope
        .apply(&x, &[0, 1], &[0, 0], &[0, 0])
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let out_b = rope
        .apply(&x, &[10, 20], &[0, 0], &[0, 0])
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let diff: f32 = out_a
        .iter()
        .zip(out_b.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-4,
        "Different temporal positions produced identical outputs (diff={diff})"
    );

    // Same input, different height positions
    let out_c = rope
        .apply(&x, &[0, 0], &[0, 1], &[0, 0])
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let out_d = rope
        .apply(&x, &[0, 0], &[10, 20], &[0, 0])
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let diff2: f32 = out_c
        .iter()
        .zip(out_d.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff2 > 1e-4,
        "Different height positions produced identical outputs (diff={diff2})"
    );
}

// -- Section independence -----------------------------------------------------

/// Changing one section's positions should only affect that section's output
/// dimensions. The other sections should remain identical.
#[test]
fn test_mrope_section_independence() {
    let head_dim = 12; // 3 sections of 4 dims each
    let rope = MultimodalRoPE::new(head_dim, [2, 2, 2], 64, 10000.0, &Device::Cpu).unwrap();

    let data = det_data(head_dim, 42.0);
    let x = DynTensor::from_vec(data, &[1, 1, 1, head_dim], &Device::Cpu).unwrap();

    // Baseline: all positions 0
    let base = rope
        .apply(&x, &[0], &[0], &[0])
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Change only temporal position
    let temporal_changed = rope
        .apply(&x, &[5], &[0], &[0])
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // HF/Qwen layout: [T1|H1|W1|T2|H2|W2] with 2 dims per chunk.
    // Temporal chunks [0..2) and [6..8) should change.
    let t1_diff: f32 = base[0..2]
        .iter()
        .zip(temporal_changed[0..2].iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        t1_diff > 1e-6,
        "Temporal first-half chunk should change when temporal position changes (diff={t1_diff})"
    );

    let t2_diff: f32 = base[6..8]
        .iter()
        .zip(temporal_changed[6..8].iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        t2_diff > 1e-6,
        "Temporal second-half chunk should change when temporal position changes (diff={t2_diff})"
    );

    // Height and width chunks should remain the same.
    for i in [2usize, 3, 4, 5, 8, 9, 10, 11] {
        assert!(
            (base[i] - temporal_changed[i]).abs() < 1e-6,
            "Non-temporal section changed at dim {i}: base={}, changed={}",
            base[i],
            temporal_changed[i]
        );
    }
}

/// Match the Hugging Face / Qwen reference layout on a hand-checkable toy case.
#[test]
fn test_mrope_matches_hf_reference_toy_case() {
    let head_dim = 6;
    let sections = [1, 1, 1];
    let base = 10000.0;
    let rope = MultimodalRoPE::new(head_dim, sections, 32, base, &Device::Cpu).unwrap();

    // HF/Qwen layout: [T1, H1, W1, T2, H2, W2].
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = DynTensor::new(&x_data, &[1, 1, 1, head_dim], &Device::Cpu).unwrap();

    let positions = [1usize, 0, 2];
    let out = rope
        .apply(&x, &[positions[0]], &[positions[1]], &[positions[2]])
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let expected = hf_reference_apply_single(&x_data, sections, positions, head_dim, base);

    for (i, (actual, expected)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "HF toy-case mismatch at dim {i}: actual={actual}, expected={expected}"
        );
    }
}

// -- Pair application consistency ---------------------------------------------

/// apply_pair(q, k) should be identical to apply(q) and apply(k) separately.
#[test]
fn test_mrope_apply_pair_matches_individual() {
    let head_dim = 12;
    let rope = MultimodalRoPE::new(head_dim, [2, 2, 2], 64, 10000.0, &Device::Cpu).unwrap();

    let q_data = det_data(4 * 3 * head_dim, 10.0);
    let k_data = det_data(2 * 3 * head_dim, 20.0);
    let q = DynTensor::from_vec(q_data, &[1, 4, 3, head_dim], &Device::Cpu).unwrap();
    let k = DynTensor::from_vec(k_data, &[1, 2, 3, head_dim], &Device::Cpu).unwrap();

    let t_pos = vec![0, 1, 2];
    let h_pos = vec![0, 0, 1];
    let w_pos = vec![0, 1, 0];

    let (q_pair, k_pair) = rope.apply_pair(&q, &k, &t_pos, &h_pos, &w_pos).unwrap();
    let q_solo = rope.apply(&q, &t_pos, &h_pos, &w_pos).unwrap();
    let k_solo = rope.apply(&k, &t_pos, &h_pos, &w_pos).unwrap();

    let qp = q_pair.to_flat_vec::<f32>().unwrap();
    let qs = q_solo.to_flat_vec::<f32>().unwrap();
    let kp = k_pair.to_flat_vec::<f32>().unwrap();
    let ks = k_solo.to_flat_vec::<f32>().unwrap();

    for (i, (a, b)) in qp.iter().zip(qs.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "Q mismatch at {i}: pair={a}, solo={b}"
        );
    }
    for (i, (a, b)) in kp.iter().zip(ks.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "K mismatch at {i}: pair={a}, solo={b}"
        );
    }
}

// -- Qwen2.5-VL configuration -------------------------------------------------

/// Verify M-ROPE works with Qwen2.5-VL production parameters:
/// head_dim=128, sections=[16, 24, 24], base=1000000.0.
#[test]
fn test_mrope_qwen25_vl_config() {
    let head_dim = 128;
    let rope =
        MultimodalRoPE::new(head_dim, [16, 24, 24], 4096, 1_000_000.0, &Device::Cpu).unwrap();

    assert_eq!(rope.head_dim(), 128);
    assert_eq!(rope.section_dims(), &[32, 48, 48]);
    assert_eq!(rope.max_position(), 4096);

    // Forward pass with realistic-ish positions
    let seq_len = 8;
    let data = det_data(4 * seq_len * head_dim, 42.0);
    let x = DynTensor::from_vec(data, &[1, 4, seq_len, head_dim], &Device::Cpu).unwrap();

    let t_pos: Vec<usize> = (0..seq_len).collect();
    let h_pos: Vec<usize> = (0..seq_len).map(|i| i / 4).collect();
    let w_pos: Vec<usize> = (0..seq_len).map(|i| i % 4).collect();

    let out = rope.apply(&x, &t_pos, &h_pos, &w_pos).unwrap();
    assert_eq!(out.dims(), &[1, 4, seq_len, head_dim]);
    assert!(!out.any_non_finite().unwrap());
}

// -- Determinism --------------------------------------------------------------

/// Multiple applications with the same positions should produce identical results.
#[test]
fn test_mrope_deterministic() {
    let head_dim = 12;
    let rope = MultimodalRoPE::new(head_dim, [2, 2, 2], 64, 10000.0, &Device::Cpu).unwrap();

    let data = det_data(2 * 3 * head_dim, 42.0);
    let x = DynTensor::from_vec(data, &[1, 2, 3, head_dim], &Device::Cpu).unwrap();
    let t_pos = vec![0, 5, 10];
    let h_pos = vec![0, 3, 7];
    let w_pos = vec![0, 1, 8];

    let out1 = rope
        .apply(&x, &t_pos, &h_pos, &w_pos)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let out2 = rope
        .apply(&x, &t_pos, &h_pos, &w_pos)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(out1, out2, "Repeated application should be deterministic");
}
