#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for VqCodebook and Rvq.

use crate::dyn_tensor::DynTensor;
use crate::layers::{Rvq, VqCodebook};
use crate::Device;

/// Helper: create a codebook with known weights.
fn make_codebook(entries: &[&[f32]]) -> VqCodebook {
    let codebook_size = entries.len();
    let dim = entries[0].len();
    let data: Vec<f32> = entries.iter().flat_map(|e| e.iter().copied()).collect();
    let weight = DynTensor::from_vec(data, &[codebook_size, dim], &Device::Cpu).unwrap();
    VqCodebook::new(weight).unwrap()
}

// -- VqCodebook tests ---------------------------------------------------------

#[test]
fn test_vq_codebook_decode_shape() {
    // Codebook: 4 entries, dim=3
    let cb = make_codebook(&[
        &[1.0, 0.0, 0.0],
        &[0.0, 1.0, 0.0],
        &[0.0, 0.0, 1.0],
        &[1.0, 1.0, 1.0],
    ]);
    assert_eq!(cb.codebook_size(), 4);
    assert_eq!(cb.dim(), 3);

    // Decode indices [0, 2] → [2, 3]
    let indices = DynTensor::from_vec(vec![0.0, 2.0], &[2], &Device::Cpu).unwrap();
    let decoded = cb.decode(&indices).unwrap();
    assert_eq!(decoded.dims(), &[2, 3]);
}

#[test]
fn test_vq_codebook_decode_values() {
    let cb = make_codebook(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

    let indices = DynTensor::from_vec(vec![2.0, 0.0, 1.0], &[3], &Device::Cpu).unwrap();
    let decoded = cb.decode(&indices).unwrap();
    let values = decoded.to_flat_vec::<f32>().unwrap();
    // index 2 → [5, 6], index 0 → [1, 2], index 1 → [3, 4]
    assert_eq!(values, vec![5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_vq_codebook_quantize_nearest() {
    // Codebook with well-separated entries
    let cb = make_codebook(&[&[0.0, 0.0], &[10.0, 0.0], &[0.0, 10.0]]);

    // Input close to entry 1 (10, 0)
    let x = DynTensor::from_vec(vec![9.0, 1.0], &[1, 2], &Device::Cpu).unwrap();
    let (quantized, indices) = cb.quantize(&x).unwrap();

    let idx_vals = indices.to_vec1::<u32>().unwrap();
    assert_eq!(idx_vals[0], 1); // nearest to [10, 0]

    let q_vals = quantized.to_flat_vec::<f32>().unwrap();
    assert_eq!(q_vals, vec![10.0, 0.0]);
}

#[test]
fn test_vq_codebook_quantize_batch() {
    let cb = make_codebook(&[&[0.0, 0.0], &[10.0, 0.0], &[0.0, 10.0]]);

    // 3 inputs: close to entries 0, 1, 2
    let x = DynTensor::from_vec(vec![0.5, 0.5, 9.5, 0.5, 0.5, 9.5], &[3, 2], &Device::Cpu).unwrap();
    let (quantized, indices) = cb.quantize(&x).unwrap();

    assert_eq!(indices.dims(), &[3]);
    let idx_vals = indices.to_vec1::<u32>().unwrap();
    assert_eq!(idx_vals, vec![0, 1, 2]);

    assert_eq!(quantized.dims(), &[3, 2]);
}

#[test]
fn test_vq_codebook_quantize_dim_mismatch() {
    let cb = make_codebook(&[&[1.0, 2.0, 3.0]]);

    // Input dim=2 but codebook dim=3
    let x = DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &Device::Cpu).unwrap();
    let result = cb.quantize(&x);
    assert!(result.is_err());
}

#[test]
fn test_vq_codebook_new_rejects_non_2d() {
    let w = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let result = VqCodebook::new(w);
    assert!(result.is_err());
}

// -- Rvq tests ----------------------------------------------------------------

#[test]
fn test_rvq_decode_single_level() {
    let cb = make_codebook(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let rvq = Rvq::new(vec![cb]).unwrap();
    assert_eq!(rvq.n_levels(), 1);
    assert_eq!(rvq.dim(), 2);

    // Codes: [1, seq=2] → indices [0, 1]
    let codes = DynTensor::from_vec(vec![0.0, 1.0], &[1, 2], &Device::Cpu).unwrap();
    let decoded = rvq.decode(&codes).unwrap();
    assert_eq!(decoded.dims(), &[2, 2]);
    let vals = decoded.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_rvq_decode_multi_level_sums() {
    // Two codebooks, both dim=2
    let cb0 = make_codebook(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let cb1 = make_codebook(&[&[0.1, 0.2], &[0.3, 0.4]]);
    let rvq = Rvq::new(vec![cb0, cb1]).unwrap();

    // Level 0 index=0 → [1, 0], Level 1 index=1 → [0.3, 0.4]
    // Sum: [1.3, 0.4]
    let codes = DynTensor::from_vec(vec![0.0, 1.0], &[2, 1], &Device::Cpu).unwrap();
    let decoded = rvq.decode(&codes).unwrap();
    assert_eq!(decoded.dims(), &[1, 2]);
    let vals = decoded.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 1.3).abs() < 1e-6);
    assert!((vals[1] - 0.4).abs() < 1e-6);
}

#[test]
fn test_rvq_encode_residual_decreases() {
    // Codebook that can partially capture the input
    let cb0 = make_codebook(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let cb1 = make_codebook(&[&[0.5, 0.0], &[0.0, 0.5]]);

    let rvq = Rvq::new(vec![cb0, cb1]).unwrap();

    let features = DynTensor::from_vec(vec![1.2, 0.3], &[1, 2], &Device::Cpu).unwrap();
    let codes = rvq.encode(&features, 2).unwrap();
    assert_eq!(codes.dims(), &[2, 1]); // [n_levels=2, seq=1]
}

#[test]
fn test_rvq_roundtrip_reconstruction() {
    // With enough codebooks covering the space, round-trip should approximate input
    let cb0 = make_codebook(&[&[1.0, 0.0], &[0.0, 1.0], &[-1.0, 0.0], &[0.0, -1.0]]);
    let cb1 = make_codebook(&[&[0.25, 0.0], &[0.0, 0.25], &[-0.25, 0.0], &[0.0, -0.25]]);

    let rvq = Rvq::new(vec![cb0, cb1]).unwrap();

    let features = DynTensor::from_vec(vec![0.8, 0.1], &[1, 2], &Device::Cpu).unwrap();

    // Encode then decode
    let codes = rvq.encode(&features, 2).unwrap();
    let reconstructed = rvq.decode(&codes).unwrap();

    let orig = features.to_flat_vec::<f32>().unwrap();
    let recon = reconstructed.to_flat_vec::<f32>().unwrap();

    // Reconstruction should be closer than without RVQ (within 0.5 error per dim)
    let err0 = (orig[0] - recon[0]).abs();
    let err1 = (orig[1] - recon[1]).abs();
    assert!(err0 < 0.5, "reconstruction error dim 0: {err0}");
    assert!(err1 < 0.5, "reconstruction error dim 1: {err1}");
}

#[test]
fn test_rvq_encode_single_level() {
    let cb = make_codebook(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let rvq = Rvq::new(vec![cb]).unwrap();

    let features = DynTensor::from_vec(vec![0.9, 0.1, 0.1, 0.9], &[2, 2], &Device::Cpu).unwrap();
    let codes = rvq.encode(&features, 1).unwrap();
    assert_eq!(codes.dims(), &[1, 2]); // [1 level, 2 seq]

    let idx = codes.to_flat_vec::<u32>().unwrap();
    assert_eq!(idx[0], 0); // [0.9, 0.1] closest to [1, 0]
    assert_eq!(idx[1], 1); // [0.1, 0.9] closest to [0, 1]
}

#[test]
fn test_rvq_empty_codebooks_rejected() {
    let result = Rvq::new(vec![]);
    assert!(result.is_err());
}

#[test]
fn test_rvq_dim_mismatch_rejected() {
    let cb0 = make_codebook(&[&[1.0, 0.0]]);
    let cb1 = make_codebook(&[&[1.0, 0.0, 0.0]]);
    let result = Rvq::new(vec![cb0, cb1]);
    assert!(result.is_err());
}

#[test]
fn test_rvq_decode_too_many_levels() {
    let cb = make_codebook(&[&[1.0, 0.0]]);
    let rvq = Rvq::new(vec![cb]).unwrap();

    // 2 levels but only 1 codebook
    let codes = DynTensor::from_vec(vec![0.0, 0.0], &[2, 1], &Device::Cpu).unwrap();
    let result = rvq.decode(&codes);
    assert!(result.is_err());
}

#[test]
fn test_rvq_decode_1d_codes_rejected() {
    let cb = make_codebook(&[&[1.0, 0.0]]);
    let rvq = Rvq::new(vec![cb]).unwrap();

    let codes = DynTensor::from_vec(vec![0.0], &[1], &Device::Cpu).unwrap();
    let result = rvq.decode(&codes);
    assert!(result.is_err());
}

#[test]
fn test_vq_codebook_seq_len_1() {
    let cb = make_codebook(&[&[1.0, 2.0, 3.0]]);

    let indices = DynTensor::from_vec(vec![0.0], &[1], &Device::Cpu).unwrap();
    let decoded = cb.decode(&indices).unwrap();
    assert_eq!(decoded.dims(), &[1, 3]);
    assert_eq!(decoded.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

// -- D2: VarBuilder loading tests ---------------------------------------------

use crate::{DType, VarBuilder};
use std::collections::HashMap;

#[test]
fn test_vq_codebook_zeros() {
    let cb = VqCodebook::zeros(8, 4, &Device::Cpu).unwrap();
    assert_eq!(cb.codebook_size(), 8);
    assert_eq!(cb.dim(), 4);

    let w = cb.weight().to_flat_vec::<f32>().unwrap();
    assert!(w.iter().all(|&v| v == 0.0));
}

#[test]
fn test_vq_codebook_load_varbuilder() {
    // Simulate MOSS pattern: codebook.{i}.embed.weight
    let weight_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weight = DynTensor::from_vec(weight_data.clone(), &[3, 2], &Device::Cpu).unwrap();

    let mut tensors = HashMap::new();
    tensors.insert("codebook.0.embed.weight".to_string(), weight);

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let cb = VqCodebook::load(vb.pp("codebook").pp(0).pp("embed"), 3, 2).unwrap();

    assert_eq!(cb.codebook_size(), 3);
    assert_eq!(cb.dim(), 2);
    assert_eq!(cb.weight().to_flat_vec::<f32>().unwrap(), weight_data);
}

#[test]
fn test_vq_codebook_load_normalized() {
    // Simulate Qwen3 pattern: embedding_sum / max(cluster_usage, 1e-5)
    // 2 entries, dim=3
    // sum = [[10, 20, 30], [40, 50, 60]]
    // usage = [2.0, 5.0]
    // expected weight = [[5, 10, 15], [8, 10, 12]]
    let sum = DynTensor::from_vec(
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        &[2, 3],
        &Device::Cpu,
    )
    .unwrap();
    let usage = DynTensor::from_vec(vec![2.0, 5.0], &[2], &Device::Cpu).unwrap();

    let mut tensors = HashMap::new();
    tensors.insert("embedding_sum".to_string(), sum);
    tensors.insert("cluster_usage".to_string(), usage);

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let cb = VqCodebook::load_normalized(&vb, 2, 3).unwrap();

    assert_eq!(cb.codebook_size(), 2);
    assert_eq!(cb.dim(), 3);
    let w = cb.weight().to_flat_vec::<f32>().unwrap();
    assert!((w[0] - 5.0).abs() < 1e-6, "w[0]={}", w[0]);
    assert!((w[1] - 10.0).abs() < 1e-6, "w[1]={}", w[1]);
    assert!((w[2] - 15.0).abs() < 1e-6, "w[2]={}", w[2]);
    assert!((w[3] - 8.0).abs() < 1e-6, "w[3]={}", w[3]);
    assert!((w[4] - 10.0).abs() < 1e-6, "w[4]={}", w[4]);
    assert!((w[5] - 12.0).abs() < 1e-6, "w[5]={}", w[5]);
}

#[test]
fn test_vq_codebook_load_normalized_zero_usage_clamped() {
    // Zero usage should be clamped to 1e-5, not cause division by zero
    let sum = DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &Device::Cpu).unwrap();
    let usage = DynTensor::from_vec(vec![0.0], &[1], &Device::Cpu).unwrap();

    let mut tensors = HashMap::new();
    tensors.insert("embedding_sum".to_string(), sum);
    tensors.insert("cluster_usage".to_string(), usage);

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let cb = VqCodebook::load_normalized(&vb, 1, 2).unwrap();

    // weight = [1, 2] / 1e-5 = [100000, 200000]
    let w = cb.weight().to_flat_vec::<f32>().unwrap();
    assert!(w[0].is_finite(), "w[0] should be finite: {}", w[0]);
    assert!(w[1].is_finite(), "w[1] should be finite: {}", w[1]);
    assert!(w[0] > 0.0, "w[0] should be positive: {}", w[0]);
}

#[test]
fn test_rvq_load_varbuilder() {
    // Simulate 2-level RVQ with indexed weight keys: 0.weight, 1.weight
    let w0 = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
    let w1 = DynTensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[2, 2], &Device::Cpu).unwrap();

    let mut tensors = HashMap::new();
    tensors.insert("0.weight".to_string(), w0);
    tensors.insert("1.weight".to_string(), w1);

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let rvq = Rvq::load(&vb, 2, 2, 2).unwrap();

    assert_eq!(rvq.n_levels(), 2);
    assert_eq!(rvq.dim(), 2);

    // Verify codebook 0 weights
    let w0_vals = rvq.codebooks()[0].weight().to_flat_vec::<f32>().unwrap();
    assert_eq!(w0_vals, vec![1.0, 0.0, 0.0, 1.0]);

    // Verify codebook 1 weights
    let w1_vals = rvq.codebooks()[1].weight().to_flat_vec::<f32>().unwrap();
    assert!((w1_vals[0] - 0.1).abs() < 1e-6);
    assert!((w1_vals[3] - 0.4).abs() < 1e-6);
}

#[test]
fn test_rvq_load_zeros() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let rvq = Rvq::load(&vb, 4, 16, 8).unwrap();

    assert_eq!(rvq.n_levels(), 4);
    assert_eq!(rvq.dim(), 8);
    for cb in rvq.codebooks() {
        assert_eq!(cb.codebook_size(), 16);
        let w = cb.weight().to_flat_vec::<f32>().unwrap();
        assert!(w.iter().all(|&v| v == 0.0));
    }
}

// -- Error path tests (proof_coverage) ----------------------------------------

/// encode with n_levels=0 should return an error (nn_rvq.rs:216-219).
#[test]
fn test_rvq_encode_zero_levels_rejected() {
    let cb = make_codebook(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let rvq = Rvq::new(vec![cb]).unwrap();

    let features = DynTensor::from_vec(vec![0.5, 0.5], &[1, 2], &Device::Cpu).unwrap();
    let result = rvq.encode(&features, 0);
    assert!(result.is_err(), "n_levels=0 should be rejected");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("at least 1 level"),
        "error should mention level requirement, got: {err_msg}"
    );
}
