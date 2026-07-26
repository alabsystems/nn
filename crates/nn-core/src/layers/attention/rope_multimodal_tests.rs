// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for MultimodalRoPE (M-ROPE).

use super::MultimodalRoPE;
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

#[test]
fn test_mrope_basic_construction() {
    let rope = MultimodalRoPE::new(12, [2, 2, 2], 64, 10000.0, &Device::Cpu).unwrap();
    assert_eq!(rope.head_dim(), 12);
    assert_eq!(rope.section_dims(), &[4, 4, 4]);
    assert_eq!(rope.max_position(), 64);
}

#[test]
fn test_mrope_asymmetric_sections() {
    let rope = MultimodalRoPE::new(128, [16, 24, 24], 4096, 1000000.0, &Device::Cpu).unwrap();
    assert_eq!(rope.head_dim(), 128);
    assert_eq!(rope.section_dims(), &[32, 48, 48]);
}

#[test]
fn test_mrope_section_sum_mismatch() {
    let err = MultimodalRoPE::new(12, [2, 2, 1], 64, 10000.0, &Device::Cpu).unwrap_err();
    assert!(format!("{err:?}").contains("sum of mrope_section_sizes"));
}

#[test]
fn test_mrope_zero_section() {
    let err = MultimodalRoPE::new(12, [0, 3, 3], 64, 10000.0, &Device::Cpu).unwrap_err();
    assert!(format!("{err:?}").contains("each section size must be > 0"));
}

#[test]
fn test_mrope_apply_preserves_shape() {
    let rope = MultimodalRoPE::new(12, [2, 2, 2], 64, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[1, 2, 4, 12], DType::F32, &Device::Cpu).unwrap();
    let out = rope
        .apply(&x, &[0, 1, 2, 3], &[0, 0, 1, 1], &[0, 1, 0, 1])
        .unwrap();
    assert_eq!(out.dims(), &[1, 2, 4, 12]);
}

#[test]
fn test_mrope_apply_pair_preserves_shapes() {
    let rope = MultimodalRoPE::new(12, [2, 2, 2], 64, 10000.0, &Device::Cpu).unwrap();
    let q = DynTensor::ones(&[1, 4, 3, 12], DType::F32, &Device::Cpu).unwrap();
    let k = DynTensor::ones(&[1, 2, 3, 12], DType::F32, &Device::Cpu).unwrap();
    let (q_rot, k_rot) = rope
        .apply_pair(&q, &k, &[0, 1, 2], &[0, 0, 1], &[0, 1, 0])
        .unwrap();
    assert_eq!(q_rot.dims(), &[1, 4, 3, 12]);
    assert_eq!(k_rot.dims(), &[1, 2, 3, 12]);
}

#[test]
fn test_mrope_text_mode_all_positions_equal() {
    let rope = MultimodalRoPE::new(12, [2, 2, 2], 64, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[1, 2, 4, 12], DType::F32, &Device::Cpu).unwrap();
    let positions = vec![0, 1, 2, 3];
    let out = rope.apply(&x, &positions, &positions, &positions).unwrap();
    assert_eq!(out.dims(), &[1, 2, 4, 12]);
    assert!(!out.any_non_finite().unwrap());
}

#[test]
fn test_mrope_position_out_of_range() {
    let rope = MultimodalRoPE::new(12, [2, 2, 2], 4, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[1, 2, 2, 12], DType::F32, &Device::Cpu).unwrap();
    let err = rope.apply(&x, &[0, 5], &[0, 0], &[0, 0]).unwrap_err();
    assert!(format!("{err:?}").contains("temporal position exceeds max_position"));
}

#[test]
fn test_mrope_position_zero_produces_identity() {
    let rope = MultimodalRoPE::new(12, [2, 2, 2], 64, 10000.0, &Device::Cpu).unwrap();
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
fn test_mrope_rank2_input() {
    let rope = MultimodalRoPE::new(12, [2, 2, 2], 64, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[3, 12], DType::F32, &Device::Cpu).unwrap();
    let out = rope.apply(&x, &[0, 1, 2], &[0, 0, 1], &[0, 1, 0]).unwrap();
    assert_eq!(out.dims(), &[3, 12]);
}

#[test]
fn test_mrope_wrong_head_dim() {
    let rope = MultimodalRoPE::new(12, [2, 2, 2], 64, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[1, 2, 3, 16], DType::F32, &Device::Cpu).unwrap();
    let err = rope
        .apply(&x, &[0, 1, 2], &[0, 0, 1], &[0, 1, 0])
        .unwrap_err();
    assert!(format!("{err:?}").contains("ShapeMismatch"));
}

#[test]
fn test_mrope_seq_len_mismatch() {
    let rope = MultimodalRoPE::new(12, [2, 2, 2], 64, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[1, 2, 3, 12], DType::F32, &Device::Cpu).unwrap();
    let err = rope.apply(&x, &[0, 1, 2], &[0, 0], &[0, 1, 0]).unwrap_err();
    assert!(format!("{err:?}").contains("DataLengthMismatch"));
}
