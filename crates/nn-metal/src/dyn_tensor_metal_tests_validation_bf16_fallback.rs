#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! BF16 raw-kernel CPU fallback tests (#1668, #1671, #1672).
//!
//! Extracted from `dyn_tensor_metal_tests_validation.rs` to keep the
//! parent file under 500 lines.
//!
//! Raw MSL kernels (gather, topk, scatter, cumsum, argreduce,
//! repeat_interleave) use hardcoded `float*` buffer types. BF16 tensors
//! must fall back to CPU instead of hard-erroring with DtypeMismatch.
//! These tests verify correct results via CPU fallback when GPU tensors
//! are BF16.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::Module;
use nn_core::{DType, Device};

use crate::test_common::init;

/// Helper: create a GPU DynTensor with BF16 dtype and specific f32 values.
fn make_bf16_gpu_tensor_with_values(data: &[f32], shape: &[usize]) -> DynTensor {
    let cpu = DynTensor::new(data, shape, &Device::Cpu).unwrap();
    let bf16_cpu = cpu.to_dtype(DType::BF16).unwrap();
    bf16_cpu.to_device(&Device::metal()).unwrap()
}

// -- BF16 raw-kernel CPU fallback tests (#1668) --------------------------------

#[test]
fn test_bf16_gather_falls_back_to_cpu() {
    init();
    let x = make_bf16_gpu_tensor_with_values(&[10.0, 20.0, 30.0, 40.0], &[4]);
    let ids = DynTensor::from_vec_u32(vec![2, 0], &[2], &Device::Cpu)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let result = x.gather(&ids, 0).unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 30.0).abs() < 0.5);
    assert!((vals[1] - 10.0).abs() < 0.5);
}

#[test]
fn test_bf16_cumsum_falls_back_to_cpu() {
    init();
    let x = make_bf16_gpu_tensor_with_values(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let result = x.cumsum(0).unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 1.0).abs() < 0.5);
    assert!((vals[1] - 3.0).abs() < 0.5);
    assert!((vals[2] - 6.0).abs() < 0.5);
    assert!((vals[3] - 10.0).abs() < 0.5);
}

#[test]
fn test_bf16_scatter_add_falls_back_to_cpu() {
    init();
    let target = make_bf16_gpu_tensor_with_values(&[0.0, 0.0, 0.0], &[3]);
    let src = make_bf16_gpu_tensor_with_values(&[1.0, 2.0, 3.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![0, 1, 2], &[3], &Device::Cpu)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let result = target.scatter_add(0, &index, &src).unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 1.0).abs() < 0.5);
    assert!((vals[1] - 2.0).abs() < 0.5);
    assert!((vals[2] - 3.0).abs() < 0.5);
}

#[test]
fn test_bf16_topk_falls_back_to_cpu() {
    init();
    let x = make_bf16_gpu_tensor_with_values(&[3.0, 1.0, 4.0, 1.0, 5.0], &[1, 5]);
    let (values, _indices) = x.topk(1, 2).unwrap();
    assert_eq!(values.dtype(), DType::BF16);
    let vals = values.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 5.0).abs() < 0.5);
    assert!((vals[1] - 4.0).abs() < 0.5);
}

#[test]
fn test_bf16_argmax_falls_back_to_cpu() {
    init();
    let x = make_bf16_gpu_tensor_with_values(&[1.0, 3.0, 2.0], &[1, 3]);
    let result = x.argmax(1).unwrap();
    let vals = result.to_flat_vec::<u32>().unwrap();
    assert_eq!(vals[0], 1);
}

#[test]
fn test_bf16_repeat_interleave_falls_back_to_cpu() {
    init();
    let x = make_bf16_gpu_tensor_with_values(&[10.0, 20.0, 30.0], &[3]);
    let repeats = DynTensor::new(&[2.0, 1.0, 3.0], &[3], &Device::Cpu).unwrap();
    let result = x.repeat_interleave(0, &repeats).unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 6);
    assert!((vals[0] - 10.0).abs() < 0.5);
    assert!((vals[1] - 10.0).abs() < 0.5);
    assert!((vals[2] - 20.0).abs() < 0.5);
    assert!((vals[3] - 30.0).abs() < 0.5);
}

// -- BF16/F16 GPU-native slice_set (#1711) ------------------------------------
//
// gpu_slice_set uses byte-level copies that handle any element width.
// BF16 and F16 tensors stay on GPU without CPU round-trip.

#[test]
fn test_bf16_slice_set_stays_on_gpu() {
    init();
    let dst = make_bf16_gpu_tensor_with_values(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
    let src = make_bf16_gpu_tensor_with_values(&[10.0, 20.0], &[2]);
    let result = dst.slice_set(0, 1, &src).unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    assert!(result.device().is_gpu(), "result should stay on GPU");
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 1.0).abs() < 0.5, "dst[0] preserved");
    assert!((vals[1] - 10.0).abs() < 0.5, "dst[1] replaced by src[0]");
    assert!((vals[2] - 20.0).abs() < 0.5, "dst[2] replaced by src[1]");
    assert!((vals[3] - 4.0).abs() < 0.5, "dst[3] preserved");
    assert!((vals[4] - 5.0).abs() < 0.5, "dst[4] preserved");
}

#[test]
fn test_bf16_slice_set_2d_dim1() {
    init();
    // [2, 4] dst, write [2, 2] src at offset 1 along dim 1
    let dst = make_bf16_gpu_tensor_with_values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    let src = make_bf16_gpu_tensor_with_values(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    let result = dst.slice_set(1, 1, &src).unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    assert!(result.device().is_gpu(), "result should stay on GPU");
    let vals = result.to_flat_vec::<f32>().unwrap();
    // Row 0: [1, 10, 20, 4]
    assert!((vals[0] - 1.0).abs() < 0.5);
    assert!((vals[1] - 10.0).abs() < 0.5);
    assert!((vals[2] - 20.0).abs() < 0.5);
    assert!((vals[3] - 4.0).abs() < 0.5);
    // Row 1: [5, 30, 40, 8]
    assert!((vals[4] - 5.0).abs() < 0.5);
    assert!((vals[5] - 30.0).abs() < 0.5);
    assert!((vals[6] - 40.0).abs() < 0.5);
    assert!((vals[7] - 8.0).abs() < 0.5);
}

/// Helper: create a GPU DynTensor with F16 dtype and specific f32 values.
fn make_f16_gpu_tensor_with_values(data: &[f32], shape: &[usize]) -> DynTensor {
    let cpu = DynTensor::new(data, shape, &Device::Cpu).unwrap();
    let f16_cpu = cpu.to_dtype(DType::F16).unwrap();
    f16_cpu.to_device(&Device::metal()).unwrap()
}

#[test]
fn test_f16_slice_set_stays_on_gpu() {
    init();
    let dst = make_f16_gpu_tensor_with_values(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let src = make_f16_gpu_tensor_with_values(&[10.0, 20.0], &[2]);
    let result = dst.slice_set(0, 1, &src).unwrap();
    assert_eq!(result.dtype(), DType::F16);
    assert!(result.device().is_gpu(), "result should stay on GPU");
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 1.0).abs() < 0.5, "dst[0] preserved");
    assert!((vals[1] - 10.0).abs() < 0.5, "dst[1] replaced by src[0]");
    assert!((vals[2] - 20.0).abs() < 0.5, "dst[2] replaced by src[1]");
    assert!((vals[3] - 4.0).abs() < 0.5, "dst[3] preserved");
}

#[test]
fn test_bf16_slice_set_kv_cache_pattern() {
    init();
    // Simulates KV cache append: [B=1, H=2, S=4, D=3] buffer, write new
    // tokens [1,2,1,3] at offset 2 along seq dim (dim=2).
    let dst_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let dst = make_bf16_gpu_tensor_with_values(&dst_data, &[1, 2, 4, 3]);
    // New 2 tokens: [1, 2, 2, 3] shape
    let src_data: Vec<f32> = (100..112).map(|i| i as f32).collect();
    let src = make_bf16_gpu_tensor_with_values(&src_data, &[1, 2, 2, 3]);
    let result = dst.slice_set(2, 2, &src).unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    assert!(result.device().is_gpu(), "result should stay on GPU");
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 24);
    // Head 0: positions 0-1 preserved, positions 2-3 overwritten
    // Offset in flat layout: head0 starts at 0, seq=2 starts at elem 6
    assert!((vals[6] - 100.0).abs() < 0.5, "h0 s2 d0 = src[0]");
    assert!((vals[7] - 101.0).abs() < 0.5, "h0 s2 d1 = src[1]");
    assert!((vals[8] - 102.0).abs() < 0.5, "h0 s2 d2 = src[2]");
    assert!((vals[9] - 103.0).abs() < 0.5, "h0 s3 d0 = src[3]");
    // First preserved elements
    assert!((vals[0] - 0.0).abs() < 0.5, "h0 s0 d0 preserved");
    assert!((vals[1] - 1.0).abs() < 0.5, "h0 s0 d1 preserved");
}

// -- BF16 GPU norm with dtype promotion (#1672, #1699) -------------------------
//
// BF16 GPU tensors are promoted to F32, run through the fused GPU norm kernel,
// and cast back to BF16 (#1699). This replaces the CPU round-trip fallback
// where recip() finiteness checks rejected near-zero variance.

#[test]
fn test_bf16_layer_norm_gpu_promoted() {
    init();
    let x = make_bf16_gpu_tensor_with_values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let weight_cpu = DynTensor::new(&[1.0, 1.0, 1.0], &[3], &Device::Cpu).unwrap();
    let bias_cpu = DynTensor::new(&[0.0, 0.0, 0.0], &[3], &Device::Cpu).unwrap();
    let weight = weight_cpu
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let bias = bias_cpu
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let result = nn_core::layers::LayerNorm::new(weight, bias, 1e-5)
        .unwrap()
        .forward(&x)
        .unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // Layer norm normalizes per-row: mean=2, std~0.816 → first elem ~ -1.22
    assert!(
        vals[0] < 0.0,
        "layer_norm[0] should be negative: {}",
        vals[0]
    );
}

#[test]
fn test_bf16_rms_norm_gpu_promoted() {
    init();
    let x = make_bf16_gpu_tensor_with_values(&[1.0, 2.0, 3.0], &[1, 3]);
    let weight_cpu = DynTensor::new(&[1.0, 1.0, 1.0], &[3], &Device::Cpu).unwrap();
    let weight = weight_cpu
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let result = nn_core::layers::RmsNorm::new(weight, 1e-5)
        .unwrap()
        .forward(&x)
        .unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // RMS norm: x / rms(x). rms([1,2,3]) ≈ 2.16 → first ≈ 0.46
    assert!(vals[0] > 0.0 && vals[0] < 1.0, "rms_norm[0]={}", vals[0]);
}

#[test]
fn test_bf16_group_norm_gpu_promoted() {
    init();
    // [batch=1, channels=4, spatial=3], num_groups=2 → 2 channels/group
    let x = make_bf16_gpu_tensor_with_values(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[1, 4, 3],
    );
    let weight_cpu = DynTensor::new(&[1.0, 1.0, 1.0, 1.0], &[4], &Device::Cpu).unwrap();
    let bias_cpu = DynTensor::new(&[0.0, 0.0, 0.0, 0.0], &[4], &Device::Cpu).unwrap();
    let weight = weight_cpu
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let bias = bias_cpu
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let result = nn_core::layers::GroupNorm::new(2, 4, weight, bias, 1e-5)
        .unwrap()
        .forward(&x)
        .unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 12);
    // Group norm normalizes within each group. Group 0 has channels [0,1]
    // with values [1,2,3,4,5,6], mean=3.5, so first elem should be negative.
    assert!(
        vals[0] < 0.0,
        "group_norm[0] should be negative: {}",
        vals[0]
    );
}

// -- BF16 GPU norm near-zero variance regression tests (#1699) ----------------
//
// dvoice Kokoro pipeline with BF16 weights hit "recip produced N non-finite
// value(s)" because CpuRoundTrip path's recip() finiteness check rejects
// near-zero variance. The GPU fused kernel path uses rsqrt without per-op
// finiteness checks, avoiding this error.

#[test]
fn test_bf16_layer_norm_near_zero_variance_no_error() {
    init();
    // Near-constant input: variance ≈ 0, which caused Inf in recip() on CPU.
    let x = make_bf16_gpu_tensor_with_values(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 3]);
    let weight_cpu = DynTensor::new(&[1.0, 1.0, 1.0], &[3], &Device::Cpu).unwrap();
    let bias_cpu = DynTensor::new(&[0.0, 0.0, 0.0], &[3], &Device::Cpu).unwrap();
    let weight = weight_cpu
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let bias = bias_cpu
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let result = nn_core::layers::LayerNorm::new(weight, bias, 1e-5)
        .unwrap()
        .forward(&x)
        .unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // Near-constant input normalized → all near zero
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "layer_norm[{i}] should be finite, got {v}");
        assert!(v.abs() < 1.0, "layer_norm[{i}]={v}, expected near 0");
    }
}

#[test]
fn test_bf16_rms_norm_near_zero_variance_no_error() {
    init();
    // Very small values: rms is tiny, rsqrt produces large values.
    // CPU recip() would reject these as Inf.
    let x = make_bf16_gpu_tensor_with_values(&[1e-4, 1e-4, 1e-4], &[1, 3]);
    let weight_cpu = DynTensor::new(&[1.0, 1.0, 1.0], &[3], &Device::Cpu).unwrap();
    let weight = weight_cpu
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let result = nn_core::layers::RmsNorm::new(weight, 1e-5)
        .unwrap()
        .forward(&x)
        .unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "rms_norm[{i}] should be finite, got {v}");
    }
}

#[test]
fn test_bf16_group_norm_near_zero_variance_no_error() {
    init();
    // Near-constant input per group — same variance issue.
    let x = make_bf16_gpu_tensor_with_values(
        &[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        &[1, 4, 3],
    );
    let weight_cpu = DynTensor::new(&[1.0, 1.0, 1.0, 1.0], &[4], &Device::Cpu).unwrap();
    let bias_cpu = DynTensor::new(&[0.0, 0.0, 0.0, 0.0], &[4], &Device::Cpu).unwrap();
    let weight = weight_cpu
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let bias = bias_cpu
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let result = nn_core::layers::GroupNorm::new(2, 4, weight, bias, 1e-5)
        .unwrap()
        .forward(&x)
        .unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "group_norm[{i}] should be finite, got {v}");
        assert!(v.abs() < 1.0, "group_norm[{i}]={v}, expected near 0");
    }
}

// -- BF16 GPU InstanceNorm fused kernel promotion (#2040) ----------------------

#[test]
fn test_bf16_instance_norm_gpu_promoted() {
    init();
    // [batch=1, channels=2, spatial=4]
    let x =
        make_bf16_gpu_tensor_with_values(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], &[1, 2, 4]);
    let result = nn_core::layers::InstanceNorm::new(1e-5)
        .unwrap()
        .forward(&x)
        .unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 8);
    // InstanceNorm normalizes per-channel: channel 0 has values [1,2,3,4]
    // with mean=2.5, so first elem should be negative (below mean).
    assert!(
        vals[0] < 0.0,
        "instance_norm[0] should be negative: {}",
        vals[0]
    );
}

#[test]
fn test_bf16_instance_norm_near_zero_variance_no_error() {
    init();
    // Near-constant input: variance ≈ 0. Fused kernel uses rsqrt(var + eps)
    // which avoids the Inf from recip() that the decomposed CPU path hits.
    let x =
        make_bf16_gpu_tensor_with_values(&[5.0, 5.0, 5.0, 5.0, -3.0, -3.0, -3.0, -3.0], &[1, 2, 4]);
    let result = nn_core::layers::InstanceNorm::new(1e-5)
        .unwrap()
        .forward(&x)
        .unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.is_finite(),
            "instance_norm[{i}] should be finite, got {v}"
        );
        assert!(v.abs() < 1.0, "instance_norm[{i}]={v}, expected near 0");
    }
}
