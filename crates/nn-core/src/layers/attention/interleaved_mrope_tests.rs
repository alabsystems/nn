// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for InterleavedMRoPE.

use super::{InterleavedMRoPE, InterleavedMRoPEConfig};
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

fn det_data(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.017).sin() * 0.5)
        .collect()
}

// -- Construction tests -------------------------------------------------------

#[test]
fn test_interleaved_mrope_basic_construction() {
    let config = InterleavedMRoPEConfig {
        head_dim: 12,
        max_position: 64,
        base: 10000.0,
    };
    let rope = InterleavedMRoPE::new(config, &Device::Cpu).unwrap();
    assert_eq!(rope.head_dim(), 12);
    assert_eq!(rope.pairs_per_section(), 2);
    assert_eq!(rope.max_position(), 64);
}

#[test]
fn test_interleaved_mrope_larger_head_dim() {
    let config = InterleavedMRoPEConfig {
        head_dim: 96,
        max_position: 4096,
        base: 1_000_000.0,
    };
    let rope = InterleavedMRoPE::new(config, &Device::Cpu).unwrap();
    assert_eq!(rope.head_dim(), 96);
    assert_eq!(rope.pairs_per_section(), 16);
}

#[test]
fn test_interleaved_mrope_invalid_head_dim_not_multiple_of_6() {
    let config = InterleavedMRoPEConfig {
        head_dim: 8,
        max_position: 64,
        base: 10000.0,
    };
    let err = InterleavedMRoPE::new(config, &Device::Cpu).unwrap_err();
    assert!(format!("{err:?}").contains("multiple of 6"));
}

#[test]
fn test_interleaved_mrope_zero_head_dim() {
    let config = InterleavedMRoPEConfig {
        head_dim: 0,
        max_position: 64,
        base: 10000.0,
    };
    let err = InterleavedMRoPE::new(config, &Device::Cpu).unwrap_err();
    assert!(format!("{err:?}").contains("multiple of 6"));
}

#[test]
fn test_interleaved_mrope_zero_max_position() {
    let config = InterleavedMRoPEConfig {
        head_dim: 12,
        max_position: 0,
        base: 10000.0,
    };
    let err = InterleavedMRoPE::new(config, &Device::Cpu).unwrap_err();
    assert!(format!("{err:?}").contains("max_position must be > 0"));
}

#[test]
fn test_interleaved_mrope_invalid_base() {
    let config = InterleavedMRoPEConfig {
        head_dim: 12,
        max_position: 64,
        base: -1.0,
    };
    let err = InterleavedMRoPE::new(config, &Device::Cpu).unwrap_err();
    assert!(format!("{err:?}").contains("base must be positive"));
}

// -- Shape preservation -------------------------------------------------------

#[test]
fn test_interleaved_mrope_apply_preserves_shape() {
    let config = InterleavedMRoPEConfig {
        head_dim: 12,
        max_position: 64,
        base: 10000.0,
    };
    let rope = InterleavedMRoPE::new(config, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[1, 2, 4, 12], DType::F32, &Device::Cpu).unwrap();
    let out = rope
        .apply(&x, &[0, 1, 2, 3], &[0, 0, 1, 1], &[0, 1, 0, 1])
        .unwrap();
    assert_eq!(out.dims(), &[1, 2, 4, 12]);
}

#[test]
fn test_interleaved_mrope_apply_pair_preserves_shapes() {
    let config = InterleavedMRoPEConfig {
        head_dim: 12,
        max_position: 64,
        base: 10000.0,
    };
    let rope = InterleavedMRoPE::new(config, &Device::Cpu).unwrap();
    let q = DynTensor::ones(&[1, 4, 3, 12], DType::F32, &Device::Cpu).unwrap();
    let k = DynTensor::ones(&[1, 2, 3, 12], DType::F32, &Device::Cpu).unwrap();
    let (q_rot, k_rot) = rope
        .apply_pair(&q, &k, &[0, 1, 2], &[0, 0, 1], &[0, 1, 0])
        .unwrap();
    assert_eq!(q_rot.dims(), &[1, 4, 3, 12]);
    assert_eq!(k_rot.dims(), &[1, 2, 3, 12]);
}

#[test]
fn test_interleaved_mrope_rank2_input() {
    let config = InterleavedMRoPEConfig {
        head_dim: 12,
        max_position: 64,
        base: 10000.0,
    };
    let rope = InterleavedMRoPE::new(config, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[3, 12], DType::F32, &Device::Cpu).unwrap();
    let out = rope.apply(&x, &[0, 1, 2], &[0, 0, 1], &[0, 1, 0]).unwrap();
    assert_eq!(out.dims(), &[3, 12]);
}

// -- Numerical properties -----------------------------------------------------

#[test]
fn test_interleaved_mrope_position_zero_is_identity() {
    let config = InterleavedMRoPEConfig {
        head_dim: 12,
        max_position: 64,
        base: 10000.0,
    };
    let rope = InterleavedMRoPE::new(config, &Device::Cpu).unwrap();
    let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let x = DynTensor::new(&data, &[1, 1, 1, 12], &Device::Cpu).unwrap();
    let out = rope.apply(&x, &[0], &[0], &[0]).unwrap();
    let out_data = out.to_flat_vec::<f32>().unwrap();
    for (a, b) in data.iter().zip(out_data.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "position 0 should be identity: {a} vs {b}"
        );
    }
}

#[test]
fn test_interleaved_mrope_preserves_l2_norm() {
    let head_dim = 24;
    let config = InterleavedMRoPEConfig {
        head_dim,
        max_position: 128,
        base: 10000.0,
    };
    let rope = InterleavedMRoPE::new(config, &Device::Cpu).unwrap();

    let data = det_data(4 * head_dim, 42.0);
    let x = DynTensor::from_vec(data, &[1, 1, 4, head_dim], &Device::Cpu).unwrap();

    let rotated = rope
        .apply(&x, &[0, 5, 10, 50], &[0, 3, 7, 12], &[0, 1, 8, 20])
        .unwrap();

    let orig = x.to_flat_vec::<f32>().unwrap();
    let rot = rotated.to_flat_vec::<f32>().unwrap();

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

#[test]
fn test_interleaved_mrope_distinct_positions_produce_distinct_outputs() {
    let head_dim = 12;
    let config = InterleavedMRoPEConfig {
        head_dim,
        max_position: 64,
        base: 10000.0,
    };
    let rope = InterleavedMRoPE::new(config, &Device::Cpu).unwrap();

    let data = det_data(2 * head_dim, 42.0);
    let x = DynTensor::from_vec(data, &[1, 1, 2, head_dim], &Device::Cpu).unwrap();

    // Different temporal positions
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

    // Different height positions
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

    // Different width positions
    let out_e = rope
        .apply(&x, &[0, 0], &[0, 0], &[0, 1])
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let out_f = rope
        .apply(&x, &[0, 0], &[0, 0], &[10, 20])
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let diff3: f32 = out_e
        .iter()
        .zip(out_f.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff3 > 1e-4,
        "Different width positions produced identical outputs (diff={diff3})"
    );
}

#[test]
fn test_interleaved_mrope_no_nan_in_output() {
    let head_dim = 12;
    let config = InterleavedMRoPEConfig {
        head_dim,
        max_position: 64,
        base: 10000.0,
    };
    let rope = InterleavedMRoPE::new(config, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[2, 4, 5, 12], DType::F32, &Device::Cpu).unwrap();
    let out = rope
        .apply(&x, &[0, 10, 20, 30, 40], &[0, 1, 2, 3, 4], &[0, 2, 4, 6, 8])
        .unwrap();
    assert!(!out.any_non_finite().unwrap());
}

// -- Interleaved vs concatenated difference -----------------------------------

/// Verify that the interleaved layout produces different results from the
/// standard HF/Qwen six-block M-ROPE (at non-zero positions), confirming the
/// interleaving is actually happening.
#[test]
fn test_interleaved_differs_from_concatenated_mrope() {
    use crate::layers::attention::MultimodalRoPE;

    let head_dim = 12;
    let config = InterleavedMRoPEConfig {
        head_dim,
        max_position: 64,
        base: 10000.0,
    };
    let interleaved = InterleavedMRoPE::new(config, &Device::Cpu).unwrap();
    let concatenated = MultimodalRoPE::new(head_dim, [2, 2, 2], 64, 10000.0, &Device::Cpu).unwrap();

    let data = det_data(2 * head_dim, 42.0);
    let x = DynTensor::from_vec(data, &[1, 1, 2, head_dim], &Device::Cpu).unwrap();

    let t_pos = vec![5, 10];
    let h_pos = vec![2, 7];
    let w_pos = vec![1, 4];

    let out_interleaved = interleaved
        .apply(&x, &t_pos, &h_pos, &w_pos)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let out_concatenated = concatenated
        .apply(&x, &t_pos, &h_pos, &w_pos)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let diff: f32 = out_interleaved
        .iter()
        .zip(out_concatenated.iter())
        .map(|(a, b)| (*a - *b).abs())
        .sum();
    assert!(
        diff > 1e-4,
        "Interleaved and standard M-ROPE should produce different results (diff={diff})"
    );
}

// -- Determinism --------------------------------------------------------------

#[test]
fn test_interleaved_mrope_deterministic() {
    let head_dim = 12;
    let config = InterleavedMRoPEConfig {
        head_dim,
        max_position: 64,
        base: 10000.0,
    };
    let rope = InterleavedMRoPE::new(config, &Device::Cpu).unwrap();

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

// -- apply_pair matches individual apply --------------------------------------

#[test]
fn test_interleaved_mrope_apply_pair_matches_individual() {
    let head_dim = 12;
    let config = InterleavedMRoPEConfig {
        head_dim,
        max_position: 64,
        base: 10000.0,
    };
    let rope = InterleavedMRoPE::new(config, &Device::Cpu).unwrap();

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

// -- Error cases --------------------------------------------------------------

#[test]
fn test_interleaved_mrope_wrong_head_dim() {
    let config = InterleavedMRoPEConfig {
        head_dim: 12,
        max_position: 64,
        base: 10000.0,
    };
    let rope = InterleavedMRoPE::new(config, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[1, 2, 3, 16], DType::F32, &Device::Cpu).unwrap();
    let err = rope
        .apply(&x, &[0, 1, 2], &[0, 0, 1], &[0, 1, 0])
        .unwrap_err();
    assert!(format!("{err:?}").contains("ShapeMismatch"));
}

#[test]
fn test_interleaved_mrope_seq_len_mismatch() {
    let config = InterleavedMRoPEConfig {
        head_dim: 12,
        max_position: 64,
        base: 10000.0,
    };
    let rope = InterleavedMRoPE::new(config, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[1, 2, 3, 12], DType::F32, &Device::Cpu).unwrap();
    let err = rope.apply(&x, &[0, 1, 2], &[0, 0], &[0, 1, 0]).unwrap_err();
    assert!(format!("{err:?}").contains("DataLengthMismatch"));
}

#[test]
fn test_interleaved_mrope_position_out_of_range() {
    let config = InterleavedMRoPEConfig {
        head_dim: 12,
        max_position: 4,
        base: 10000.0,
    };
    let rope = InterleavedMRoPE::new(config, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[1, 2, 2, 12], DType::F32, &Device::Cpu).unwrap();
    let err = rope.apply(&x, &[0, 5], &[0, 0], &[0, 0]).unwrap_err();
    assert!(format!("{err:?}").contains("temporal position exceeds max_position"));
}

#[test]
fn test_interleaved_mrope_rank1_rejected() {
    let config = InterleavedMRoPEConfig {
        head_dim: 12,
        max_position: 64,
        base: 10000.0,
    };
    let rope = InterleavedMRoPE::new(config, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[12], DType::F32, &Device::Cpu).unwrap();
    let err = rope.apply(&x, &[0], &[0], &[0]).unwrap_err();
    assert!(format!("{err:?}").contains("RankMismatch"));
}

// -- Numerical regression: hand-computed single pair --------------------------

/// Verify the rotation math for a single pair at a known position.
/// For pair index 0 (section=temporal), position=1, base=10000:
///   theta_0 = base^(-0/head_dim) = 1.0
///   angle = 1.0 * 1 = 1.0
///   cos(1.0) ~= 0.5403, sin(1.0) ~= 0.8415
///   y_even = x_even * cos - x_odd * sin
///   y_odd  = x_even * sin + x_odd * cos
#[test]
fn test_interleaved_mrope_single_pair_numerical() {
    let head_dim = 6; // Minimum: 3 sections x 1 pair each
    let config = InterleavedMRoPEConfig {
        head_dim,
        max_position: 4,
        base: 10000.0,
    };
    let rope = InterleavedMRoPE::new(config, &Device::Cpu).unwrap();

    // Input: [1, 1, 1, 6] with known values
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = DynTensor::new(&x_data, &[1, 1, 1, 6], &Device::Cpu).unwrap();

    // temporal=1, height=0, width=0
    let out = rope
        .apply(&x, &[1], &[0], &[0])
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Pair 0 (section=temporal=0, global_pair_idx=0):
    //   theta = base^(-0/6) = 1.0, angle = 1 * 1.0 = 1.0
    //   cos(1.0) = 0.5403023, sin(1.0) = 0.84147096
    let cos_t = 1.0f32.cos();
    let sin_t = 1.0f32.sin();
    let expected_0 = x_data[0] * cos_t - x_data[1] * sin_t;
    let expected_1 = x_data[0] * sin_t + x_data[1] * cos_t;

    assert!(
        (out[0] - expected_0).abs() < 1e-5,
        "pair 0 even: expected {expected_0}, got {}",
        out[0]
    );
    assert!(
        (out[1] - expected_1).abs() < 1e-5,
        "pair 0 odd: expected {expected_1}, got {}",
        out[1]
    );

    // Pair 1 (section=height=1, position=0): angle=0, cos=1, sin=0 => identity
    assert!(
        (out[2] - x_data[2]).abs() < 1e-6,
        "pair 1 even: expected identity, got {}",
        out[2]
    );
    assert!(
        (out[3] - x_data[3]).abs() < 1e-6,
        "pair 1 odd: expected identity, got {}",
        out[3]
    );

    // Pair 2 (section=width=2, position=0): angle=0, cos=1, sin=0 => identity
    assert!(
        (out[4] - x_data[4]).abs() < 1e-6,
        "pair 2 even: expected identity, got {}",
        out[4]
    );
    assert!(
        (out[5] - x_data[5]).abs() < 1e-6,
        "pair 2 odd: expected identity, got {}",
        out[5]
    );
}

// -- Interleaved section assignment verification ------------------------------

/// Verify that changing only temporal position affects pairs 0, 3, 6, ...
/// (i.e., pairs where index % 3 == 0), and leaves pairs 1, 2, 4, 5, ... unchanged.
#[test]
fn test_interleaved_mrope_section_assignment() {
    let head_dim = 18; // 9 pairs: indices 0..9, sections: 0,1,2,0,1,2,0,1,2
    let config = InterleavedMRoPEConfig {
        head_dim,
        max_position: 64,
        base: 10000.0,
    };
    let rope = InterleavedMRoPE::new(config, &Device::Cpu).unwrap();

    let data = det_data(head_dim, 42.0);
    let x = DynTensor::from_vec(data, &[1, 1, 1, head_dim], &Device::Cpu).unwrap();

    // Baseline: all positions 0
    let base_out = rope
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

    // Temporal pairs (index % 3 == 0): indices 0,1, 6,7, 12,13 should differ
    for pair_idx in [0, 3, 6] {
        let i = pair_idx * 2;
        let diff = (base_out[i] - temporal_changed[i]).abs()
            + (base_out[i + 1] - temporal_changed[i + 1]).abs();
        assert!(
            diff > 1e-6,
            "Temporal pair {pair_idx} (dims {i},{}) should change when temporal position changes (diff={diff})",
            i + 1
        );
    }

    // Height pairs (index % 3 == 1): indices 2,3, 8,9, 14,15 should NOT change
    for pair_idx in [1, 4, 7] {
        let i = pair_idx * 2;
        assert!(
            (base_out[i] - temporal_changed[i]).abs() < 1e-6,
            "Height pair {pair_idx} dim {i} should NOT change"
        );
        assert!(
            (base_out[i + 1] - temporal_changed[i + 1]).abs() < 1e-6,
            "Height pair {pair_idx} dim {} should NOT change",
            i + 1
        );
    }

    // Width pairs (index % 3 == 2): indices 4,5, 10,11, 16,17 should NOT change
    for pair_idx in [2, 5, 8] {
        let i = pair_idx * 2;
        assert!(
            (base_out[i] - temporal_changed[i]).abs() < 1e-6,
            "Width pair {pair_idx} dim {i} should NOT change"
        );
        assert!(
            (base_out[i + 1] - temporal_changed[i + 1]).abs() < 1e-6,
            "Width pair {pair_idx} dim {} should NOT change",
            i + 1
        );
    }
}
