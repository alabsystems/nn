// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for under-tested nn layers.
//!
//! Targets layers with zero or low test counts: ConvTranspose1d, GroupNorm,
//! WeightNormConv1d, NanCheckPolicy, SDPA/causal_mask/repeat_kv,
//! RmsNorm (extended), Sequential (extended), Dropout (extended),
//! Embedding (edge cases), InstanceNorm (extended), Linear (extended),
//! LayerNorm (extended), RotaryEmbedding.

#![allow(deprecated)]

use crate::dyn_tensor::DynTensor;
use crate::layers::*;
use crate::{DType, Device};

fn cpu() -> Device {
    Device::Cpu
}

// ---------------------------------------------------------------------------
// ConvTranspose1d — 0 dedicated tests before this file
// ---------------------------------------------------------------------------

#[test]
fn test_conv_transpose1d_config_builder() {
    let cfg = ConvTranspose1dConfig::new(1, 2, 1)
        .with_output_padding(1)
        .with_groups(2);
    assert_eq!(cfg.padding, 1);
    assert_eq!(cfg.stride, 2);
    assert_eq!(cfg.dilation, 1);
    assert_eq!(cfg.output_padding, 1);
    assert_eq!(cfg.groups, 2);
}

#[test]
fn test_conv_transpose1d_weight_rank_error() {
    let w = DynTensor::from_vec(vec![1.0; 4], &[2, 2], &cpu()).unwrap();
    let err = ConvTranspose1d::new(w, None, ConvTranspose1dConfig::default());
    assert!(err.is_err());
}

#[test]
fn test_conv_transpose1d_groups_zero_error() {
    let w = DynTensor::from_vec(vec![1.0; 3], &[1, 1, 3], &cpu()).unwrap();
    let cfg = ConvTranspose1dConfig::default().with_groups(0);
    let err = ConvTranspose1d::new(w, None, cfg);
    assert!(err.is_err());
}

#[test]
fn test_conv_transpose1d_accessors() {
    let w = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![0.5], &[1], &cpu()).unwrap();
    let layer = ConvTranspose1d::new(w, Some(b), ConvTranspose1dConfig::default()).unwrap();
    assert_eq!(layer.weight().dims(), &[1, 1, 3]);
    assert!(layer.bias().is_some());
    assert_eq!(layer.config().stride, 1);
}

#[test]
fn test_conv_transpose1d_no_bias_identity_kernel() {
    // kernel=[1] with stride=1 should be identity
    let w = DynTensor::from_vec(vec![1.0], &[1, 1, 1], &cpu()).unwrap();
    let layer = ConvTranspose1d::new(w, None, ConvTranspose1dConfig::default()).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4], &cpu()).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4]);
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_conv_transpose1d_output_finite() {
    let w = DynTensor::from_vec(vec![0.5, -0.5, 0.25], &[1, 1, 3], &cpu()).unwrap();
    let layer = ConvTranspose1d::new(w, None, ConvTranspose1dConfig::default()).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4], &cpu()).unwrap();
    let y = layer.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!(v.is_finite(), "ConvTranspose1d output must be finite");
    }
}

#[test]
fn test_conv_transpose1d_batch_dim() {
    let w = DynTensor::from_vec(vec![1.0, 1.0], &[1, 1, 2], &cpu()).unwrap();
    let layer = ConvTranspose1d::new(w, None, ConvTranspose1dConfig::default()).unwrap();
    // Batch of 2
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 1, 2], &cpu()).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims()[0], 2);
    assert_eq!(y.dims()[1], 1);
}

// ---------------------------------------------------------------------------
// GroupNorm — 0 inline tests (only in tests_norm.rs basic)
// ---------------------------------------------------------------------------

#[test]
fn test_group_norm_basic_forward() {
    let w = DynTensor::ones(&[4], DType::F32, &cpu()).unwrap();
    let b = DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap();
    let gn = GroupNorm::new(2, 4, w, b, 1e-5).unwrap();
    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 4, 2],
        &cpu(),
    )
    .unwrap();
    let y = gn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 4, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!(v.is_finite(), "GroupNorm output must be finite");
    }
}

#[test]
fn test_group_norm_single_group() {
    // num_groups = 1 is global normalization (all channels in one group)
    let w = DynTensor::ones(&[2], DType::F32, &cpu()).unwrap();
    let b = DynTensor::zeros(&[2], DType::F32, &cpu()).unwrap();
    let gn = GroupNorm::new(1, 2, w, b, 1e-5).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 3.0, 5.0, 7.0], &[1, 2, 2], &cpu()).unwrap();
    let y = gn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // With single group, all 4 values normalized together: mean=4, std=sqrt(5)
    // (1-4)/sqrt(5+eps) ≈ -1.3416
    assert!((vals[0] - (-1.3416)).abs() < 0.01);
}

#[test]
fn test_group_norm_zero_groups_error() {
    let w = DynTensor::ones(&[4], DType::F32, &cpu()).unwrap();
    let b = DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap();
    assert!(GroupNorm::new(0, 4, w, b, 1e-5).is_err());
}

#[test]
fn test_group_norm_weight_shape_mismatch() {
    let w = DynTensor::ones(&[3], DType::F32, &cpu()).unwrap();
    let b = DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap();
    assert!(GroupNorm::new(2, 4, w, b, 1e-5).is_err());
}

#[test]
fn test_group_norm_accessors() {
    let w = DynTensor::ones(&[4], DType::F32, &cpu()).unwrap();
    let b = DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap();
    let gn = GroupNorm::new(2, 4, w, b, 1e-5).unwrap();
    assert_eq!(gn.weight().dims(), &[4]);
    assert_eq!(gn.bias().dims(), &[4]);
}

#[test]
fn test_group_norm_batch_dim() {
    let w = DynTensor::ones(&[2], DType::F32, &cpu()).unwrap();
    let b = DynTensor::zeros(&[2], DType::F32, &cpu()).unwrap();
    let gn = GroupNorm::new(1, 2, w, b, 1e-5).unwrap();
    // Batch of 3
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[3, 2, 2], &cpu()).unwrap();
    let y = gn.forward(&x).unwrap();
    assert_eq!(y.dims()[0], 3);
}

#[test]
fn test_group_norm_rank_error() {
    let w = DynTensor::ones(&[4], DType::F32, &cpu()).unwrap();
    let b = DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap();
    let gn = GroupNorm::new(2, 4, w, b, 1e-5).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], &cpu()).unwrap();
    assert!(gn.forward(&x).is_err());
}

// ---------------------------------------------------------------------------
// WeightNormConv1d — 0 inline tests
// ---------------------------------------------------------------------------

#[test]
fn test_weight_norm_conv1d_normalization() {
    // v: [1, 1, 3] = [3, 4, 0], ||v|| = 5
    // g: [1, 1, 1] = [10]
    // normalized = 10/5 * [3, 4, 0] = [6, 8, 0]
    let v = DynTensor::from_vec(vec![3.0, 4.0, 0.0], &[1, 1, 3], &cpu()).unwrap();
    let g = DynTensor::from_vec(vec![10.0], &[1, 1, 1], &cpu()).unwrap();
    let cfg = Conv1dConfig::new(1, 1, 1); // padding=1
    let layer = WeightNormConv1d::new(v, g, None, cfg).unwrap();
    let x = DynTensor::from_vec(vec![0.0, 1.0, 0.0], &[1, 1, 3], &cpu()).unwrap();
    let y = layer.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // With padding=1, input is [0, 0, 1, 0, 0]
    // kernel = [6, 8, 0]
    // conv output at pos 1: 0*6 + 1*8 + 0*0 = 8.0
    assert!((vals[1] - 8.0).abs() < 0.01);
}

#[test]
fn test_weight_norm_conv1d_with_bias() {
    let v = DynTensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 1, 3], &cpu()).unwrap();
    let g = DynTensor::from_vec(vec![1.0], &[1, 1, 1], &cpu()).unwrap();
    let bias = DynTensor::from_vec(vec![5.0], &[1], &cpu()).unwrap();
    let cfg = Conv1dConfig::new(1, 1, 1);
    let layer = WeightNormConv1d::new(v, g, Some(bias), cfg).unwrap();
    let x = DynTensor::from_vec(vec![2.0, 3.0, 4.0], &[1, 1, 3], &cpu()).unwrap();
    let y = layer.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // All outputs should include +5 bias
    for v in &vals {
        assert!(*v >= 5.0 - 0.1, "bias should shift output by +5");
    }
}

#[test]
fn test_weight_norm_conv1d_output_finite() {
    let v = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let g = DynTensor::from_vec(vec![1.0], &[1, 1, 1], &cpu()).unwrap();
    let layer = WeightNormConv1d::new(v, g, None, Conv1dConfig::default()).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &cpu()).unwrap();
    let y = layer.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!(v.is_finite());
    }
}

// ---------------------------------------------------------------------------
// NanCheckPolicy — 0 tests
// ---------------------------------------------------------------------------

#[test]
fn test_nan_check_policy_default_is_always() {
    assert_eq!(nan_check_policy(), NanCheckPolicy::Always);
}

#[test]
fn test_nan_check_policy_skip_scope() {
    assert_eq!(nan_check_policy(), NanCheckPolicy::Always);
    with_nan_check_policy(NanCheckPolicy::Skip, || {
        assert_eq!(nan_check_policy(), NanCheckPolicy::Skip);
    });
    // Restored after scope exits
    assert_eq!(nan_check_policy(), NanCheckPolicy::Always);
}

#[test]
fn test_nan_check_policy_nested_scopes() {
    with_nan_check_policy(NanCheckPolicy::Skip, || {
        assert_eq!(nan_check_policy(), NanCheckPolicy::Skip);
        with_nan_check_policy(NanCheckPolicy::Always, || {
            assert_eq!(nan_check_policy(), NanCheckPolicy::Always);
        });
        assert_eq!(nan_check_policy(), NanCheckPolicy::Skip);
    });
    assert_eq!(nan_check_policy(), NanCheckPolicy::Always);
}

#[test]
fn test_check_output_finite_passes_for_finite() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    assert!(check_output_finite(&t, "test_layer").is_ok());
}

#[test]
fn test_check_output_finite_fails_for_nan() {
    let t = DynTensor::from_vec(vec![1.0, f32::NAN, 3.0], &[3], &cpu()).unwrap();
    assert!(check_output_finite(&t, "test_layer").is_err());
}

#[test]
fn test_check_output_finite_fails_for_inf() {
    let t = DynTensor::from_vec(vec![f32::INFINITY, 2.0], &[2], &cpu()).unwrap();
    assert!(check_output_finite(&t, "test_layer").is_err());
}

#[test]
fn test_check_output_finite_skip_policy_passes_for_nan() {
    let t = DynTensor::from_vec(vec![f32::NAN], &[1], &cpu()).unwrap();
    with_nan_check_policy(NanCheckPolicy::Skip, || {
        assert!(check_output_finite(&t, "test_layer").is_ok());
    });
}

// ---------------------------------------------------------------------------
// SDPA — 0 non-kani tests in sdpa.rs
// ---------------------------------------------------------------------------

#[test]
fn test_sdpa_basic_no_mask() {
    // [1, 1, 2, 4] => B=1, H=1, S=2, D=4
    let q = DynTensor::from_vec(
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        &[1, 1, 2, 4],
        &cpu(),
    )
    .unwrap();
    let k = q.clone();
    let v = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 1, 2, 4],
        &cpu(),
    )
    .unwrap();
    let out = sdpa(&q, &k, &v, None, 1.0).unwrap();
    assert_eq!(out.dims(), &[1, 1, 2, 4]);
    let vals = out.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!(v.is_finite());
    }
}

#[test]
fn test_sdpa_scale_invariant() {
    let q = DynTensor::from_vec(vec![1.0; 4], &[1, 1, 1, 4], &cpu()).unwrap();
    let k = q.clone();
    let v = DynTensor::from_vec(vec![2.0; 4], &[1, 1, 1, 4], &cpu()).unwrap();
    // Single-token self-attention: output should be v regardless of scale
    let out = sdpa(&q, &k, &v, None, 0.5).unwrap();
    let vals = out.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!((*v - 2.0).abs() < 1e-5, "single-token SDPA should return V");
    }
}

#[test]
fn test_sdpa_non_finite_scale_error() {
    let q = DynTensor::from_vec(vec![1.0; 4], &[1, 1, 1, 4], &cpu()).unwrap();
    let k = q.clone();
    let v = q.clone();
    assert!(sdpa(&q, &k, &v, None, f64::NAN).is_err());
    assert!(sdpa(&q, &k, &v, None, f64::INFINITY).is_err());
}

// ---------------------------------------------------------------------------
// causal_mask — 0 non-kani tests
// ---------------------------------------------------------------------------

#[test]
fn test_causal_mask_shape() {
    let mask = causal_mask(4, &cpu()).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 4, 4]);
}

#[test]
fn test_causal_mask_values() {
    let mask = causal_mask(3, &cpu()).unwrap();
    let vals = mask.to_flat_vec::<f32>().unwrap();
    // Row 0: [0, -inf, -inf] -> can attend to position 0 only
    assert_eq!(vals[0], 0.0);
    assert_eq!(vals[1], f32::NEG_INFINITY);
    assert_eq!(vals[2], f32::NEG_INFINITY);
    // Row 1: [0, 0, -inf] -> can attend to 0,1
    assert_eq!(vals[3], 0.0);
    assert_eq!(vals[4], 0.0);
    assert_eq!(vals[5], f32::NEG_INFINITY);
    // Row 2: [0, 0, 0] -> can attend to all
    assert_eq!(vals[6], 0.0);
    assert_eq!(vals[7], 0.0);
    assert_eq!(vals[8], 0.0);
}

#[test]
fn test_causal_mask_with_offset_shape() {
    let mask = causal_mask_with_offset(2, 5, DType::F32, &cpu()).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 2, 5]);
}

#[test]
fn test_causal_mask_zero_seq_len_error() {
    assert!(causal_mask_with_offset(0, 5, DType::F32, &cpu()).is_err());
    assert!(causal_mask_with_offset(5, 0, DType::F32, &cpu()).is_err());
}

#[test]
fn test_causal_mask_total_less_than_new_error() {
    assert!(causal_mask_with_offset(5, 3, DType::F32, &cpu()).is_err());
}

// ---------------------------------------------------------------------------
// repeat_kv — 0 non-kani tests
// ---------------------------------------------------------------------------

#[test]
fn test_repeat_kv_no_repeat() {
    let x = DynTensor::from_vec(vec![1.0; 8], &[1, 2, 2, 2], &cpu()).unwrap();
    let y = repeat_kv(&x, 1).unwrap();
    assert_eq!(y.dims(), &[1, 2, 2, 2]);
}

#[test]
fn test_repeat_kv_doubles_heads() {
    let x = DynTensor::from_vec(vec![1.0; 4], &[1, 1, 2, 2], &cpu()).unwrap();
    let y = repeat_kv(&x, 2).unwrap();
    assert_eq!(y.dims(), &[1, 2, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 8);
    // Both heads should have the same values (repeated from 1 head)
    assert_eq!(&vals[0..4], &vals[4..8]);
}

// ---------------------------------------------------------------------------
// RmsNorm — extend from 3 tests
// ---------------------------------------------------------------------------

#[test]
fn test_rms_norm_batch_dim() {
    let weight = DynTensor::ones(&[4], DType::F32, &cpu()).unwrap();
    let norm = RmsNorm::new(weight, 1e-5).unwrap();
    let data: Vec<f32> = (0..16).map(|i| i as f32 + 1.0).collect();
    let input = DynTensor::from_vec(data, &[2, 2, 4], &cpu()).unwrap();
    let output = norm.forward(&input).unwrap();
    assert_eq!(output.dims(), &[2, 2, 4]);
}

#[test]
fn test_rms_norm_constant_input() {
    let weight = DynTensor::ones(&[3], DType::F32, &cpu()).unwrap();
    let norm = RmsNorm::new(weight, 1e-5).unwrap();
    let input = DynTensor::from_vec(vec![5.0, 5.0, 5.0], &[1, 3], &cpu()).unwrap();
    let output = norm.forward(&input).unwrap();
    let vals = output.to_flat_vec::<f32>().unwrap();
    // RMS of [5,5,5] = 5, normed = [1,1,1]
    for v in &vals {
        assert!(
            (*v - 1.0).abs() < 0.01,
            "constant input should normalize to ~1"
        );
    }
}

#[test]
fn test_rms_norm_invalid_eps() {
    let weight = DynTensor::ones(&[4], DType::F32, &cpu()).unwrap();
    assert!(RmsNorm::new(weight.clone(), -1.0).is_err());
    assert!(RmsNorm::new(weight.clone(), f64::NAN).is_err());
    assert!(RmsNorm::new(weight, f64::INFINITY).is_err());
}

#[test]
fn test_rms_norm_weight_must_be_1d() {
    let weight = DynTensor::ones(&[2, 3], DType::F32, &cpu()).unwrap();
    assert!(RmsNorm::new(weight, 1e-5).is_err());
}

#[test]
fn test_rms_norm_accessor() {
    let weight = DynTensor::ones(&[8], DType::F32, &cpu()).unwrap();
    let norm = RmsNorm::new(weight, 1e-6).unwrap();
    assert_eq!(norm.weight().dims(), &[8]);
}

// ---------------------------------------------------------------------------
// Sequential — extend from 3 tests
// ---------------------------------------------------------------------------

#[test]
fn test_sequential_default() {
    let seq = Sequential::default();
    assert!(seq.is_empty());
    assert_eq!(seq.len(), 0);
}

#[test]
fn test_sequential_chain_multiple() {
    let mut seq = Sequential::new();
    seq.add_fn(super::super::dyn_tensor::DynTensor::relu);
    seq.add_fn(super::super::dyn_tensor::DynTensor::sqr);
    seq.add_fn(DynTensor::neg);
    assert_eq!(seq.len(), 3);
    let x = DynTensor::from_vec(vec![-2.0, 3.0], &[2], &cpu()).unwrap();
    let y = seq.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // relu(-2,3) = (0,3), sqr = (0,9), neg = (0,-9)
    assert_eq!(vals, vec![0.0, -9.0]);
}

// ---------------------------------------------------------------------------
// Dropout — extend from 4 tests
// ---------------------------------------------------------------------------

#[test]
fn test_dropout_forward_exact_values_preserved() {
    let d = Dropout::new(0.9);
    let vals = vec![1.5, -2.3, 0.0, 42.0];
    let x = DynTensor::from_vec(vals.clone(), &[4], &cpu()).unwrap();
    let y = d.forward(&x).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vals);
}

#[test]
fn test_dropout_3d_input() {
    let d = Dropout::new(0.5);
    let x = DynTensor::from_vec(vec![1.0; 24], &[2, 3, 4], &cpu()).unwrap();
    let y = d.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4]);
}

// ---------------------------------------------------------------------------
// Embedding — extend coverage of edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_embedding_forward_ids_basic() {
    let w = DynTensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0], &[3, 2], &cpu()).unwrap();
    let emb = Embedding::new(w).unwrap();
    let result = emb.forward_ids(&[0, 2]).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 0.0, 2.0, 2.0]);
}

#[test]
fn test_embedding_forward_ids_out_of_range() {
    let w = DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &cpu()).unwrap();
    let emb = Embedding::new(w).unwrap();
    assert!(emb.forward_ids(&[1]).is_err()); // vocab_size=1, index=1 is OOB
}

#[test]
fn test_embedding_forward_u32_tensor() {
    let w = DynTensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2], &cpu()).unwrap();
    let emb = Embedding::new(w).unwrap();
    let ids = DynTensor::from_vec_u32(vec![1, 0], &[2], &cpu()).unwrap();
    let result = emb.forward(&ids).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![30.0, 40.0, 10.0, 20.0]);
}

#[test]
fn test_embedding_weight_rank_error() {
    let w = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    assert!(Embedding::new(w).is_err());
}

#[test]
fn test_embedding_accessors() {
    let w = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let emb = Embedding::new(w).unwrap();
    assert_eq!(emb.weight().dims(), &[2, 2]);
    assert_eq!(emb.embeddings().dims(), &[2, 2]);
}

// ---------------------------------------------------------------------------
// InstanceNorm — extend (only tested in instance_norm_tests.rs)
// ---------------------------------------------------------------------------

#[test]
fn test_instance_norm_basic() {
    let inorm = InstanceNorm::new(1e-5).unwrap();
    // [B=1, C=1, T=4]
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4], &cpu()).unwrap();
    let y = inorm.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // mean=2.5, std=sqrt(1.25), normed approx [-1.342, -0.447, 0.447, 1.342]
    assert!(vals[0] < 0.0, "first value should be negative");
    assert!(vals[3] > 0.0, "last value should be positive");
}

#[test]
fn test_instance_norm_rank_error() {
    let inorm = InstanceNorm::new(1e-5).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    assert!(inorm.forward(&x).is_err());
    let x2 = DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &cpu()).unwrap();
    assert!(inorm.forward(&x2).is_err());
}

#[test]
fn test_instance_norm_eps_validation() {
    assert!(InstanceNorm::new(-1.0).is_err());
    assert!(InstanceNorm::new(f64::NAN).is_err());
}

#[test]
fn test_instance_norm_precision_mode() {
    let inorm = InstanceNorm::with_precision(1e-5, InstanceNormPrecision::MatchPyTorchCpu).unwrap();
    assert_eq!(inorm.eps(), 1e-5);
    let x = DynTensor::from_vec(vec![1.0, 3.0, 5.0, 7.0], &[1, 1, 4], &cpu()).unwrap();
    let y = inorm.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4]);
}

// ---------------------------------------------------------------------------
// Linear — extend (tested in tests.rs but missing edge cases)
// ---------------------------------------------------------------------------

#[test]
fn test_linear_rank_error() {
    let w = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    assert!(Linear::new(w, None).is_err());
}

#[test]
fn test_linear_bias_shape_mismatch() {
    let w = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let bad_bias = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    assert!(Linear::new(w, Some(bad_bias)).is_err());
}

#[test]
fn test_linear_accessors() {
    let w = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![0.0, 0.0], &[2], &cpu()).unwrap();
    let lin = Linear::new(w, Some(b)).unwrap();
    assert_eq!(lin.in_features(), 2);
    assert_eq!(lin.out_features(), 2);
    assert!(lin.bias().is_some());
}

#[test]
fn test_linear_3d_input() {
    // Linear should work on 3D input [B, S, D]
    let w = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &cpu()).unwrap();
    let lin = Linear::new(w, None).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2], &cpu()).unwrap();
    let y = lin.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2, 2]);
}

// ---------------------------------------------------------------------------
// LayerNorm — extend (tested in tests_norm.rs but missing edge cases)
// ---------------------------------------------------------------------------

#[test]
fn test_layer_norm_weight_bias_mismatch() {
    let w = DynTensor::from_vec(vec![1.0, 1.0], &[2], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![0.0, 0.0, 0.0], &[3], &cpu()).unwrap();
    assert!(LayerNorm::new(w, b, 1e-5).is_err());
}

#[test]
fn test_layer_norm_invalid_eps() {
    let w = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![0.0], &[1], &cpu()).unwrap();
    assert!(LayerNorm::new(w.clone(), b.clone(), -1.0).is_err());
    assert!(LayerNorm::new(w, b, f64::NAN).is_err());
}

#[test]
fn test_layer_norm_3d_input() {
    let w = DynTensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4], &cpu()).unwrap();
    let b = DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap();
    let ln = LayerNorm::new(w, b, 1e-5).unwrap();
    let x = DynTensor::from_vec((0..24).map(|i| i as f32).collect(), &[2, 3, 4], &cpu()).unwrap();
    let y = ln.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!(v.is_finite());
    }
}

#[test]
fn test_layer_norm_accessors() {
    let w = DynTensor::from_vec(vec![1.0, 1.0], &[2], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![0.0, 0.0], &[2], &cpu()).unwrap();
    let ln = LayerNorm::new(w, b, 1e-5).unwrap();
    assert_eq!(ln.weight().dims(), &[2]);
    assert_eq!(ln.bias().dims(), &[2]);
}

// ---------------------------------------------------------------------------
// RotaryEmbedding — tested in rope_tests.rs but sdpa.rs has no tests
// ---------------------------------------------------------------------------

#[test]
fn test_rotary_embedding_creation() {
    let rope = RotaryEmbedding::new(8, 32, 10000.0, &cpu()).unwrap();
    // apply to [B=1, H=1, S=4, D=8]
    let x = DynTensor::from_vec(vec![1.0; 32], &[1, 1, 4, 8], &cpu()).unwrap();
    let y = rope.apply(&x, 0).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4, 8]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!(v.is_finite());
    }
}

#[test]
fn test_rotary_embedding_odd_head_dim_error() {
    assert!(RotaryEmbedding::new(7, 32, 10000.0, &cpu()).is_err());
}

#[test]
fn test_rotary_embedding_zero_head_dim_error() {
    assert!(RotaryEmbedding::new(0, 32, 10000.0, &cpu()).is_err());
}

#[test]
fn test_rotary_embedding_zero_max_seq_len_error() {
    assert!(RotaryEmbedding::new(8, 0, 10000.0, &cpu()).is_err());
}

// ---------------------------------------------------------------------------
// Pooling layers — extend coverage (pool_tests.rs has 12 tests)
// ---------------------------------------------------------------------------

#[test]
fn test_max_pool1d_basic() {
    let pool = MaxPool1d::new(Pool1dConfig::new(2)).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 3.0, 2.0, 4.0], &[1, 1, 4], &cpu()).unwrap();
    let y = pool.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![3.0, 4.0]);
}

#[test]
fn test_max_pool1d_zero_kernel_error() {
    assert!(MaxPool1d::new(Pool1dConfig::new(0)).is_err());
}

#[test]
fn test_max_pool2d_zero_kernel_error() {
    assert!(MaxPool2d::new(Pool2dConfig::new(0)).is_err());
}

#[test]
fn test_adaptive_avg_pool2d_zero_output_error() {
    assert!(AdaptiveAvgPool2d::new(0, 1).is_err());
    assert!(AdaptiveAvgPool2d::new(1, 0).is_err());
}

#[test]
fn test_adaptive_avg_pool2d_output_size() {
    let pool = AdaptiveAvgPool2d::new(2, 3).unwrap();
    assert_eq!(pool.output_size(), (2, 3));
}

// ---------------------------------------------------------------------------
// Activation — already has 10 tests but verify comprehensive coverage
// ---------------------------------------------------------------------------

#[test]
fn test_activation_relu_negative() {
    let x = DynTensor::from_vec(vec![-1.0, 0.0, 1.0], &[3], &cpu()).unwrap();
    let y = Activation::Relu.forward(&x).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![0.0, 0.0, 1.0]);
}

#[test]
fn test_activation_sigmoid_known() {
    let x = DynTensor::from_vec(vec![0.0], &[1], &cpu()).unwrap();
    let y = Activation::Sigmoid.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 0.5).abs() < 1e-5);
}
